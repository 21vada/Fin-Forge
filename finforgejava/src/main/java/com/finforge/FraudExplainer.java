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

public class FraudExplainer {

    private static final Path CSV_PATH = Path.of(
            "C:/Users/pc/Documents/fin forge/ieee-fraud-detection/train_transaction.csv"
    );
    private static final Path MODELS_DIR = Path.of("models");

    private static final int TRAIN_ROWS = 413_378;
    private static final int VALIDATION_ROWS = 88_581;

    private static final String[] NUMERIC_FEATURES = {
            "TransactionAmt", "TransactionDT", "dist1",
            "C1", "C2", "C3", "C5", "C6", "C9", "C11", "C13",
            "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10"
    };
    private static final String[] PRODUCTS = { "W", "C", "H", "S", "R" };
    private static final String[] CATEGORICAL_FEATURES = {
            "P_emaildomain", "R_emaildomain", "card4", "card6", "M4"
    };

    public static void main(String[] args) throws Exception {

        System.out.println("=== FINFORGE EXPLANATION SERVICE ===");

        Path modelPath = MODELS_DIR.resolve("xgboost.model.json");
        if (!Files.exists(modelPath)) {
            System.err.println("Model not found. Run FraudXGBoost first.");
            return;
        }

        System.out.println("Loading saved XGBoost model...");
        Booster model = XGBoost.loadModel(modelPath.toString());

        int bestRoundCount = Integer.parseInt(
                Files.readString(MODELS_DIR.resolve("xgboost_best_round_count.txt")).trim()
        );

        String[] featureNames = createFeatureNames();

        System.out.println("Rebuilding categorical encoders from training data...");
        Map<String, Map<String, Float>> categoryEncoders = buildEncoders();

        System.out.println("Scanning test set for a known fraud case...");
        TransactionRecord targetTx = findFirstTestFraud(categoryEncoders);

        if (targetTx == null) {
            System.err.println("Could not find a fraud case in the test set.");
            return;
        }

        System.out.println("\nAnalyzing Transaction ID: " + targetTx.transactionId);
        System.out.println("Amount: $" + targetTx.amount + " | Product: " + targetTx.product);

        // Build a 1-row DMatrix for the target transaction
        DMatrix matrix = new DMatrix(targetTx.encodedFeatures, 1, featureNames.length, Float.NaN);
        matrix.setFeatureNames(featureNames);

        // Extract raw probabilities and feature contributions
        float[][] predictions = model.predict(matrix, false, bestRoundCount);
        float[][] contributions = model.predictContrib(matrix, bestRoundCount);

        double probability = predictions[0][0];
        System.out.printf(Locale.US, "XGBoost Predicted Risk Score: %.2f%%%n%n", probability * 100);

        // The contribution array has size [featureCount + 1]. The last element is the base bias.
        float[] rowContribs = contributions[0];
        float baseMargin = rowContribs[featureNames.length];

        System.out.println("EVIDENCE BREAKDOWN (Feature Contributions)");
        System.out.println("--------------------------------------------------");
        System.out.printf(Locale.US, "%-25s | %10s%n", "[Base Baseline Risk]", String.format("%+.4f", baseMargin));

        List<FeatureContribution> evidence = new ArrayList<>();
        for (int i = 0; i < featureNames.length; i++) {
            if (rowContribs[i] != 0.0f) {
                evidence.add(new FeatureContribution(featureNames[i], rowContribs[i], targetTx.rawValues.get(featureNames[i])));
            }
        }

        // Sort by absolute impact (highest impact first)
        evidence.sort(Comparator.comparingDouble((FeatureContribution c) -> Math.abs(c.contribution)).reversed());

        for (int i = 0; i < Math.min(10, evidence.size()); i++) {
            FeatureContribution ev = evidence.get(i);
            String rawValue = ev.rawValue == null || ev.rawValue.isBlank() ? "(missing)" : ev.rawValue;
            System.out.printf(Locale.US, "%-25s | %10s | Value: %s%n",
                    ev.featureName, String.format("%+.4f", ev.contribution), rawValue);
        }

        System.out.println("--------------------------------------------------");
        System.out.println("Note: Positive contributions increase fraud probability. Negative contributions decrease it.");

        model.dispose();
        matrix.dispose();
    }

    private static Map<String, Map<String, Float>> buildEncoders() throws IOException {
        Map<String, Map<String, Float>> encoders = new HashMap<>();
        for (String cat : CATEGORICAL_FEATURES) encoders.put(cat, new HashMap<>());

        long rowIndex = 0;
        try (
                Reader reader = Files.newBufferedReader(CSV_PATH, StandardCharsets.UTF_8);
                CSVParser parser = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).get().parse(reader)
        ) {
            for (CSVRecord row : parser) {
                if (rowIndex >= TRAIN_ROWS) break;
                for (String cat : CATEGORICAL_FEATURES) {
                    String val = row.get(cat);
                    if (val != null && !val.isBlank()) {
                        Map<String, Float> dict = encoders.get(cat);
                        dict.computeIfAbsent(val, k -> (float) dict.size() + 1f);
                    }
                }
                rowIndex++;
            }
        }
        return encoders;
    }

    private static TransactionRecord findFirstTestFraud(Map<String, Map<String, Float>> encoders) throws IOException {
        long rowIndex = 0;
        long targetStart = TRAIN_ROWS + VALIDATION_ROWS;

        try (
                Reader reader = Files.newBufferedReader(CSV_PATH, StandardCharsets.UTF_8);
                CSVParser parser = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).get().parse(reader)
        ) {
            for (CSVRecord row : parser) {
                if (rowIndex >= targetStart) {
                    int label = Integer.parseInt(row.get("isFraud"));
                    if (label == 1) {
                        return buildRecord(row, encoders);
                    }
                }
                rowIndex++;
            }
        }
        return null;
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
            String featName = "ProductCD_" + category;
            rawValues.put(featName, category.equals(product) ? "1" : "0");
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

    private static class TransactionRecord {
        final String transactionId;
        final String amount;
        final String product;
        final float[] encodedFeatures;
        final Map<String, String> rawValues;

        TransactionRecord(String transactionId, String amount, String product, float[] encodedFeatures, Map<String, String> rawValues) {
            this.transactionId = transactionId;
            this.amount = amount;
            this.product = product;
            this.encodedFeatures = encodedFeatures;
            this.rawValues = rawValues;
        }
    }

    private static class FeatureContribution {
        final String featureName;
        final float contribution;
        final String rawValue;

        FeatureContribution(String featureName, float contribution, String rawValue) {
            this.featureName = featureName;
            this.contribution = contribution;
            this.rawValue = rawValue;
        }
    }
}