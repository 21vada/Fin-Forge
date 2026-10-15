package com.finforge;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public class SequenceTensorGenerator {

    private static final Path CSV_PATH = Path.of(
            "C:/Users/pc/Documents/fin forge/ieee-fraud-detection/train_transaction.csv"
    );
    private static final Path OUTPUT_PATH = Path.of("reports", "lstm_sequential_tensors.csv");

    // The LSTM "memory" length: How many past transactions should it look at?
    private static final int WINDOW_SIZE = 5;

    // We will track a simplified feature set for this tensor extraction
    private static final String[] TRACKED_FEATURES = {
            "TransactionAmt", "dist1", "C1", "C13"
    };

    public static void main(String[] args) throws IOException {
        System.out.println("=== FINFORGE TEMPORAL TENSOR GENERATOR ===");

        if (!Files.exists(CSV_PATH)) {
            System.err.println("Dataset not found: " + CSV_PATH.toAbsolutePath());
            return;
        }

        System.out.println("1. Reading dataset and grouping by Card ID (card1)...");
        Map<String, List<TxNode>> cardHistories = loadAndGroupHistories();

        System.out.println("Unique Cards Tracked: " + cardHistories.size());

        System.out.println("\n2. Sorting chronological sequences and extracting rolling windows...");
        List<TensorWindow> allWindows = extractRollingWindows(cardHistories);

        System.out.println("Generated Tensors (5-step sequences): " + allWindows.size());

        System.out.println("\n3. Exporting 3D Tensors to flattened CSV format...");
        exportTensors(allWindows);

        analyzeSequences(allWindows);
    }

    private static Map<String, List<TxNode>> loadAndGroupHistories() throws IOException {
        Map<String, List<TxNode>> histories = new HashMap<>();
        long rowsProcessed = 0;

        try (
                Reader reader = Files.newBufferedReader(CSV_PATH, StandardCharsets.UTF_8);
                CSVParser parser = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).get().parse(reader)
        ) {
            for (CSVRecord row : parser) {
                String cardId = row.get("card1");
                if (cardId == null || cardId.isBlank()) continue;

                TxNode node = new TxNode(
                        row.get("TransactionID"),
                        Long.parseLong(row.get("TransactionDT")),
                        Integer.parseInt(row.get("isFraud")),
                        extractFeatures(row)
                );

                histories.computeIfAbsent(cardId, k -> new ArrayList<>()).add(node);

                rowsProcessed++;
                if (rowsProcessed % 100_000 == 0) {
                    System.out.println(" Processed " + rowsProcessed + " rows...");
                }
            }
        }
        return histories;
    }

    private static float[] extractFeatures(CSVRecord row) {
        float[] features = new float[TRACKED_FEATURES.length];
        for (int i = 0; i < TRACKED_FEATURES.length; i++) {
            String val = row.get(TRACKED_FEATURES[i]);
            try {
                features[i] = (val == null || val.isBlank()) ? 0.0f : Float.parseFloat(val);
            } catch (NumberFormatException e) {
                features[i] = 0.0f;
            }
        }
        return features;
    }

    private static List<TensorWindow> extractRollingWindows(Map<String, List<TxNode>> histories) {
        List<TensorWindow> windows = new ArrayList<>();

        for (Map.Entry<String, List<TxNode>> entry : histories.entrySet()) {
            String cardId = entry.getKey();
            List<TxNode> sequence = entry.getValue();

            // Ignore cards with only a single transaction (no sequence behavior to learn)
            if (sequence.size() < 2) continue;

            // Sort by time to ensure the LSTM reads the events in exact order
            sequence.sort(Comparator.comparingLong(n -> n.timestamp));

            // Slide a window of size N across the card's history
            for (int i = 0; i < sequence.size(); i++) {
                // We only create a window if the CURRENT transaction is the target
                TxNode currentTarget = sequence.get(i);

                List<TxNode> window = new ArrayList<>();
                // Walk backward to gather historical context (padding if necessary)
                for (int step = WINDOW_SIZE - 1; step >= 0; step--) {
                    int historyIndex = i - step;
                    if (historyIndex < 0) {
                        window.add(null); // Zero-padding for early transactions
                    } else {
                        window.add(sequence.get(historyIndex));
                    }
                }
                windows.add(new TensorWindow(cardId, window, currentTarget.isFraud));
            }
        }
        return windows;
    }

    private static void exportTensors(List<TensorWindow> windows) throws IOException {
        Files.createDirectories(OUTPUT_PATH.getParent());

        try (
                Writer writer = Files.newBufferedWriter(OUTPUT_PATH, StandardCharsets.UTF_8);
                CSVPrinter printer = new CSVPrinter(writer, CSVFormat.DEFAULT)
        ) {
            // Build dynamic headers based on timesteps
            List<String> headers = new ArrayList<>();
            headers.add("CardID");

            for (int t = WINDOW_SIZE - 1; t >= 0; t--) {
                String prefix = (t == 0) ? "t_0_" : "t_minus_" + t + "_";
                headers.add(prefix + "TimeDelta");
                for (String f : TRACKED_FEATURES) headers.add(prefix + f);
            }
            headers.add("Target_isFraud");

            printer.printRecord(headers);

            for (TensorWindow win : windows) {
                List<Object> record = new ArrayList<>();
                record.add(win.cardId);

                TxNode currentTarget = win.steps.get(WINDOW_SIZE - 1);

                for (int step = 0; step < WINDOW_SIZE; step++) {
                    TxNode node = win.steps.get(step);
                    if (node == null) {
                        // Zero padding for empty timesteps
                        record.add(0.0f); // TimeDelta
                        for (int f = 0; f < TRACKED_FEATURES.length; f++) record.add(0.0f);
                    } else {
                        long timeDelta = currentTarget.timestamp - node.timestamp;
                        record.add(timeDelta);
                        for (float f : node.features) record.add(f);
                    }
                }
                record.add(win.targetFraud);
                printer.printRecord(record);
            }
        }
        System.out.println("Exported sequence data to: " + OUTPUT_PATH.toAbsolutePath());
    }

    private static void analyzeSequences(List<TensorWindow> windows) {
        long normalSequences = 0;
        long fraudSequences = 0;

        for (TensorWindow w : windows) {
            if (w.targetFraud == 1) fraudSequences++;
            else normalSequences++;
        }

        System.out.println("\n--- SEQUENCE ANALYSIS ---");
        System.out.printf(Locale.US, "Normal Behavioral Chains: %d%n", normalSequences);
        System.out.printf(Locale.US, "Fraudulent Behavioral Chains: %d%n", fraudSequences);
        System.out.println("The data is now shaped for an LSTM. It can look at 't_minus_4', 't_minus_3', etc., to calculate velocity before predicting 't_0'.");
    }

    private static class TxNode {
        final String txId;
        final long timestamp;
        final int isFraud;
        final float[] features;

        TxNode(String txId, long timestamp, int isFraud, float[] features) {
            this.txId = txId;
            this.timestamp = timestamp;
            this.isFraud = isFraud;
            this.features = features;
        }
    }

    private static class TensorWindow {
        final String cardId;
        final List<TxNode> steps;
        final int targetFraud;

        TensorWindow(String cardId, List<TxNode> steps, int targetFraud) {
            this.cardId = cardId;
            this.steps = steps;
            this.targetFraud = targetFraud;
        }
    }
}