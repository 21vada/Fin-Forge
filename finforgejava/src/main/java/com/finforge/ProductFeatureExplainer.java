package com.finforge;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public class ProductFeatureExplainer {

    private static final Path CSV_PATH = Path.of(
            "C:/Users/pc/Documents/fin forge/ieee-fraud-detection/train_transaction.csv"
    );
    private static final Path MODELS_DIR = Path.of("models");
    private static final Path REPORTS_DIR = Path.of("reports");

    private static final int TRAIN_ROWS = 413_378;

    private static final String[] NUMERIC_FEATURES = {
            "TransactionAmt", "TransactionDT", "dist1",
            "C1", "C2", "C3", "C5", "C6", "C9", "C11", "C13",
            "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10"
    };
    private static final String[] PRODUCTS = { "W", "C", "H", "S", "R" };
    private static final String[] CATEGORICAL_FEATURES = {
            "P_emaildomain", "R_emaildomain", "card4", "card6", "M4"
    };

    private static final Map<String, String> FEATURE_DICTIONARY = Map.of(
            "TransactionAmt", "Transaction Amount",
            "TransactionDT", "Time Delta",
            "P_emaildomain", "Purchaser Email Domain",
            "R_emaildomain", "Recipient Email Domain",
            "card4", "Card Network",
            "card6", "Card Type",
            "dist1", "Distance 1"
    );

    public static void main(String[] args) throws Exception {
        System.out.println("=== FINFORGE PRODUCT FEATURE DIAGNOSTICS ===\n");

        Path predictionsPath = REPORTS_DIR.resolve("xgboost_test_predictions.csv");
        if (!Files.exists(predictionsPath)) {
            System.err.println("Cannot find " + predictionsPath.toAbsolutePath());
            return;
        }

        // 1. Identify the highest-scoring fraud case for each product
        System.out.println("Scanning predictions to identify representative fraud cases per product...");
        Map<String, TargetTransaction> targetIds = findTopFraudPerProduct(predictionsPath);

        for (String prod : targetIds.keySet()) {
            System.out.printf("Selected Transaction %s for Product %s (Score: %.2f%%)%n",
                    targetIds.get(prod).transactionId, prod, targetIds.get(prod).probability * 100);
        }

        // 2. Build Encoders & Extract Raw Data for Targets
        System.out.println("\nBuilding encoders and extracting feature vectors...");
        Map<String, Map<String, Float>> encoders = new HashMap<>();
        for (String cat : CATEGORICAL_FEATURES) encoders.put(cat, new HashMap<>());

        List<TransactionRecord> targetRecords = extractTargetRecords(targetIds, encoders);

        // 3. Load Model and Generate Explanations
        Path modelPath = MODELS_DIR.resolve("xgboost.model.json");
        Booster model = XGBoost.loadModel(modelPath.toString());
        int bestRoundCount = Integer.parseInt(Files.readString(MODELS_DIR.resolve("xgboost_best_round_count.txt")).trim());
        String[] featureNames = createFeatureNames();

        System.out.println("\nGenerating mathematical explanations...\n");

        for (TransactionRecord tx : targetRecords) {
            DMatrix matrix = new DMatrix(tx.encodedFeatures, 1, featureNames.length, Float.NaN);
            matrix.setFeatureNames(featureNames);

            float[][] contributions = model.predictContrib(matrix, bestRoundCount);
            float[] rowContribs = contributions[0];

            List<FeatureContribution> escalators = new ArrayList<>();
            for (int i = 0; i < featureNames.length; i++) {
                if (rowContribs[i] > 0.01f) {
                    escalators.add(new FeatureContribution(featureNames[i], rowContribs[i], tx.rawValues.get(featureNames[i])));
                }
            }

            // Sort by highest positive impact
            escalators.sort(Comparator.comparingDouble((FeatureContribution c) -> c.contribution).reversed());

            System.out.println("==================================================");
            System.out.printf("PRODUCT: %s | TRANSACTION: %s | AMOUNT: $%s%n", tx.product, tx.transactionId, tx.amount);
            System.out.printf("XGBOOST PREDICTED RISK: %.2f%%%n", targetIds.get(tx.product).probability * 100);
            System.out.println("--------------------------------------------------");
            System.out.println("TOP RISK ESCALATORS (What drove the fraud flag?):");

            for (int i = 0; i < Math.min(5, escalators.size()); i++) {
                FeatureContribution ev = escalators.get(i);
                String readableName = FEATURE_DICTIONARY.getOrDefault(ev.featureName, ev.featureName);
                String value = ev.rawValue == null || ev.rawValue.isBlank() ? "Missing" : ev.rawValue;
                System.out.printf(Locale.US, " * %-25s : %-15s (+%.4f log-odds)%n", readableName, value, ev.contribution);
            }
            System.out.println("==================================================\n");

            matrix.dispose();
        }

        model.dispose();
    }

    private static Map<String, TargetTransaction> findTopFraudPerProduct(Path path) throws IOException {
        Map<String, TargetTransaction> topFrauds = new HashMap<>();

        try (
                Reader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8);
                CSVParser parser = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).get().parse(reader)
        ) {
            for (CSVRecord row : parser) {
                int actualFraud = Integer.parseInt(row.get("actualFraud"));
                if (actualFraud == 1) {
                    String product = row.get("ProductCD");
                    double prob = Double.parseDouble(row.get("fraudProbability"));

                    if (!topFrauds.containsKey(product) || prob > topFrauds.get(product).probability) {
                        topFrauds.put(product, new TargetTransaction(row.get("TransactionID"), product, prob));
                    }
                }
            }
        }
        return topFrauds;
    }

    private static List<TransactionRecord> extractTargetRecords(Map<String, TargetTransaction> targetIds, Map<String, Map<String, Float>> encoders) throws IOException {
        List<TransactionRecord> records = new ArrayList<>();
        Map<String, String> targetIdToProduct = new HashMap<>();
        for (TargetTransaction t : targetIds.values()) targetIdToProduct.put(t.transactionId, t.product);

        long rowIndex = 0;
        try (
                Reader reader = Files.newBufferedReader(CSV_PATH, StandardCharsets.UTF_8);
                CSVParser parser = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).get().parse(reader)
        ) {
            for (CSVRecord row : parser) {
                // Phase 1: Build Encoders from training data
                if (rowIndex < TRAIN_ROWS) {
                    for (String cat : CATEGORICAL_FEATURES) {
                        String val = row.get(cat);
                        if (val != null && !val.isBlank()) encoders.get(cat).computeIfAbsent(val, k -> (float) encoders.get(cat).size() + 1f);
                    }
                }

                // Phase 2: Extract targets anywhere in the file
                String txId = row.get("TransactionID");
                if (targetIdToProduct.containsKey(txId)) {
                    records.add(buildRecord(row, encoders));
                }
                rowIndex++;
            }
        }
        return records;
    }

    private static TransactionRecord buildRecord(CSVRecord row, Map<String, Map<String, Float>> encoders) {
        String[] names = createFeatureNames();
        float[] features = new float[names.length];
        Map<String, String> rawValues = new HashMap<>();
        int column = 0;

        for (String feature : NUMERIC_FEATURES) {
            String val = row.get(feature);
            rawValues.put(feature, val);
            features[column++] = parseNumber(val);
        }

        String product = row.get("ProductCD");
        for (String category : PRODUCTS) {
            rawValues.put("ProductCD_" + category, category.equals(product) ? "1" : "0");
            features[column++] = category.equals(product) ? 1.0f : 0.0f;
        }

        for (String cat : CATEGORICAL_FEATURES) {
            String val = row.get(cat);
            rawValues.put(cat, val);
            if (val == null || val.isBlank()) {
                features[column++] = Float.NaN;
            } else {
                Float id = encoders.get(cat).get(val);
                features[column++] = (id != null) ? id : Float.NaN;
            }
        }
        return new TransactionRecord(row.get("TransactionID"), row.get("TransactionAmt"), product, features, rawValues);
    }

    private static String[] createFeatureNames() {
        String[] names = new String[NUMERIC_FEATURES.length + PRODUCTS.length + CATEGORICAL_FEATURES.length];
        System.arraycopy(NUMERIC_FEATURES, 0, names, 0, NUMERIC_FEATURES.length);
        for (int i = 0; i < PRODUCTS.length; i++) names[NUMERIC_FEATURES.length + i] = "ProductCD_" + PRODUCTS[i];
        for (int i = 0; i < CATEGORICAL_FEATURES.length; i++) names[NUMERIC_FEATURES.length + PRODUCTS.length + i] = CATEGORICAL_FEATURES[i];
        return names;
    }

    private static float parseNumber(String raw) {
        if (raw == null || raw.isBlank()) return Float.NaN;
        try { return (float) Double.parseDouble(raw); } catch (NumberFormatException e) { return Float.NaN; }
    }

    private static class TargetTransaction {
        final String transactionId, product;
        final double probability;
        TargetTransaction(String transactionId, String product, double probability) {
            this.transactionId = transactionId; this.product = product; this.probability = probability;
        }
    }

    private static class TransactionRecord {
        final String transactionId, amount, product;
        final float[] encodedFeatures;
        final Map<String, String> rawValues;
        TransactionRecord(String transactionId, String amount, String product, float[] encodedFeatures, Map<String, String> rawValues) {
            this.transactionId = transactionId; this.amount = amount; this.product = product;
            this.encodedFeatures = encodedFeatures; this.rawValues = rawValues;
        }
    }

    private static class FeatureContribution {
        final String featureName, rawValue;
        final float contribution;
        FeatureContribution(String featureName, float contribution, String rawValue) {
            this.featureName = featureName; this.contribution = contribution; this.rawValue = rawValue;
        }
    }
}