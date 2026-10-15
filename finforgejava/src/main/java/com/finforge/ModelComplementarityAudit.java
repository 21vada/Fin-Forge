package com.finforge;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public class ModelComplementarityAudit {

    private static final Path RF_PREDICTIONS = Path.of(
            "reports", "random_forest_test_predictions.csv"
    );

    private static final Path XGB_PREDICTIONS = Path.of(
            "reports", "xgboost_test_predictions.csv"
    );

    public static void main(String[] args) throws IOException {

        System.out.println("=== FINFORGE MODEL COMPLEMENTARITY AUDIT ===");

        if (!Files.exists(RF_PREDICTIONS) || !Files.exists(XGB_PREDICTIONS)) {
            System.err.println("Missing prediction files. Please ensure both FraudRandomForest and FraudXGBoost have run.");
            return;
        }

        System.out.println("Loading XGBoost predictions...");
        Map<String, Prediction> xgbData = loadPredictions(XGB_PREDICTIONS);

        System.out.println("Loading Random Forest predictions...");
        Map<String, Prediction> rfData = loadPredictions(RF_PREDICTIONS);

        System.out.println("\nAligning predictions by TransactionID...");
        List<JoinedPrediction> joined = new ArrayList<>();

        for (Map.Entry<String, Prediction> entry : xgbData.entrySet()) {
            String txId = entry.getKey();
            Prediction xgb = entry.getValue();
            Prediction rf = rfData.get(txId);

            if (rf == null) {
                throw new IllegalStateException("TransactionID " + txId + " missing from Random Forest output.");
            }

            if (xgb.actualFraud != rf.actualFraud) {
                throw new IllegalStateException("Label mismatch for TransactionID " + txId);
            }

            joined.add(new JoinedPrediction(txId, xgb.actualFraud, rf, xgb));
        }

        System.out.println("Successfully aligned " + joined.size() + " test transactions.");

        // 1. Calculate overall alert agreement
        long flaggedBoth = 0;
        long flaggedRfOnly = 0;
        long flaggedXgbOnly = 0;
        long flaggedNeither = 0;

        // 2. Calculate fraud catches
        long fraudCaughtBoth = 0;
        long fraudCaughtRfOnly = 0;
        long fraudCaughtXgbOnly = 0;
        long fraudMissed = 0;

        for (JoinedPrediction jp : joined) {
            boolean rfAlert = jp.rf.predictedFraud == 1;
            boolean xgbAlert = jp.xgb.predictedFraud == 1;
            boolean isFraud = jp.actualFraud == 1;

            if (rfAlert && xgbAlert) {
                flaggedBoth++;
                if (isFraud) fraudCaughtBoth++;
            } else if (rfAlert) {
                flaggedRfOnly++;
                if (isFraud) fraudCaughtRfOnly++;
            } else if (xgbAlert) {
                flaggedXgbOnly++;
                if (isFraud) fraudCaughtXgbOnly++;
            } else {
                flaggedNeither++;
                if (isFraud) fraudMissed++;
            }
        }

        RankingMetrics rfMetrics = calculateRankingMetrics(joined, false);
        RankingMetrics xgbMetrics = calculateRankingMetrics(joined, true);

        String report = buildReport(
                joined.size(),
                flaggedBoth, flaggedRfOnly, flaggedXgbOnly, flaggedNeither,
                fraudCaughtBoth, fraudCaughtRfOnly, fraudCaughtXgbOnly, fraudMissed,
                rfMetrics, xgbMetrics
        );

        Path reportPath = Path.of("reports", "model_complementarity_audit.txt");
        Files.writeString(reportPath, report, StandardCharsets.UTF_8);

        System.out.println("\n" + report);
        System.out.println("Saved audit report to: " + reportPath.toAbsolutePath());
    }

    private static Map<String, Prediction> loadPredictions(Path path) throws IOException {
        Map<String, Prediction> map = new HashMap<>();

        try (
                Reader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8);
                CSVParser parser = CSVFormat.DEFAULT.builder()
                        .setHeader()
                        .setSkipHeaderRecord(true)
                        .get()
                        .parse(reader)
        ) {
            for (CSVRecord row : parser) {
                String txId = row.get("TransactionID");
                int actualFraud = Integer.parseInt(row.get("actualFraud"));
                double prob = Double.parseDouble(row.get("fraudProbability"));
                int predicted = Integer.parseInt(row.get("predictedFraud"));

                map.put(txId, new Prediction(actualFraud, prob, predicted));
            }
        }
        return map;
    }

    private static RankingMetrics calculateRankingMetrics(
            List<JoinedPrediction> predictions,
            boolean useXGBoost
    ) {
        ScoreLabel[] sorted = new ScoreLabel[predictions.size()];
        long positives = 0;

        for (int i = 0; i < predictions.size(); i++) {
            JoinedPrediction jp = predictions.get(i);
            double prob = useXGBoost ? jp.xgb.probability : jp.rf.probability;

            sorted[i] = new ScoreLabel(prob, jp.actualFraud);

            if (jp.actualFraud == 1) {
                positives++;
            }
        }

        long negatives = predictions.size() - positives;

        Arrays.sort(
                sorted,
                Comparator.comparingDouble((ScoreLabel x) -> x.probability).reversed()
        );

        long tp = 0;
        long fp = 0;

        double previousRecall = 0;
        double previousFpr = 0;
        double previousPrecision = 1.0;

        double rocAuc = 0;
        double prAuc = 0;

        int i = 0;

        while (i < sorted.length) {
            double currentScore = sorted[i].probability;
            int j = i;

            // Group tied probabilities to ensure identical thresholds evaluate identically
            while (j < sorted.length && Double.compare(sorted[j].probability, currentScore) == 0) {
                if (sorted[j].label == 1) {
                    tp++;
                } else {
                    fp++;
                }
                j++;
            }

            double recall = (double) tp / positives;
            double fpr = (double) fp / negatives;
            double precision = tp + fp == 0 ? 1.0 : (double) tp / (tp + fp);

            rocAuc += (fpr - previousFpr) * (recall + previousRecall) / 2.0;
            prAuc += (recall - previousRecall) * (precision + previousPrecision) / 2.0;

            previousRecall = recall;
            previousFpr = fpr;
            previousPrecision = precision;

            i = j;
        }

        return new RankingMetrics(rocAuc, prAuc);
    }

    private static String buildReport(
            int totalRows,
            long flaggedBoth, long flaggedRfOnly, long flaggedXgbOnly, long flaggedNeither,
            long fraudCaughtBoth, long fraudCaughtRfOnly, long fraudCaughtXgbOnly, long fraudMissed,
            RankingMetrics rfMetrics, RankingMetrics xgbMetrics
    ) {
        StringBuilder out = new StringBuilder();

        out.append("=== FINFORGE MODEL COMPLEMENTARITY AUDIT ===\n\n");
        out.append("Total transactions aligned: ").append(totalRows).append("\n\n");

        out.append("1. OVERALL ALERT AGREEMENT (All Transactions)\n");
        out.append("--------------------------------------------------\n");
        out.append("Flagged by BOTH:         ").append(flaggedBoth).append("\n");
        out.append("Flagged by RF ONLY:      ").append(flaggedRfOnly).append("\n");
        out.append("Flagged by XGBoost ONLY: ").append(flaggedXgbOnly).append("\n");
        out.append("Flagged by NEITHER:      ").append(flaggedNeither).append("\n\n");

        out.append("2. FRAUD DETECTION COMPLEMENTARITY (Actual Fraud Only)\n");
        out.append("--------------------------------------------------\n");
        out.append("Caught by BOTH:          ").append(fraudCaughtBoth).append("\n");
        out.append("Caught by RF ONLY:       ").append(fraudCaughtRfOnly).append("  <-- Unique RF value\n");
        out.append("Caught by XGBoost ONLY:  ").append(fraudCaughtXgbOnly).append("  <-- Unique XGB value\n");
        out.append("Missed by BOTH:          ").append(fraudMissed).append("\n\n");

        out.append("3. STANDARDIZED RANKING METRICS (Single Evaluator)\n");
        out.append("--------------------------------------------------\n");
        out.append(String.format(Locale.US, "Random Forest ROC-AUC:   %.4f%n", rfMetrics.rocAuc));
        out.append(String.format(Locale.US, "Random Forest PR-AUC:    %.4f%n", rfMetrics.prAuc));
        out.append(String.format(Locale.US, "XGBoost ROC-AUC:         %.4f%n", xgbMetrics.rocAuc));
        out.append(String.format(Locale.US, "XGBoost PR-AUC:          %.4f%n", xgbMetrics.prAuc));

        out.append("\nNote: PR-AUC calculated using trapezoidal interpolation.\n");

        return out.toString();
    }

    private static class Prediction {
        final int actualFraud;
        final double probability;
        final int predictedFraud;

        Prediction(int actualFraud, double probability, int predictedFraud) {
            this.actualFraud = actualFraud;
            this.probability = probability;
            this.predictedFraud = predictedFraud;
        }
    }

    private static class JoinedPrediction {
        final String transactionId;
        final int actualFraud;
        final Prediction rf;
        final Prediction xgb;

        JoinedPrediction(String transactionId, int actualFraud, Prediction rf, Prediction xgb) {
            this.transactionId = transactionId;
            this.actualFraud = actualFraud;
            this.rf = rf;
            this.xgb = xgb;
        }
    }

    private static class ScoreLabel {
        final double probability;
        final int label;

        ScoreLabel(double probability, int label) {
            this.probability = probability;
            this.label = label;
        }
    }

    private static class RankingMetrics {
        final double rocAuc;
        final double prAuc;

        RankingMetrics(double rocAuc, double prAuc) {
            this.rocAuc = rocAuc;
            this.prAuc = prAuc;
        }
    }
}