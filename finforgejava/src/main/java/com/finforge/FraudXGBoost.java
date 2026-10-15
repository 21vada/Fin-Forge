package com.finforge;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import java.util.Locale;

public class FraudXGBoost {

    private static final Path CSV_PATH = Path.of(
            "C:/Users/pc/Documents/fin forge/ieee-fraud-detection/train_transaction.csv"
    );

    private static final int TRAIN_ROWS = 413_378;
    private static final int VALIDATION_ROWS = 88_581;
    private static final int TEST_ROWS = 88_581;
    private static final int TOTAL_ROWS =
            TRAIN_ROWS + VALIDATION_ROWS + TEST_ROWS;

    private static final String[] NUMERIC_FEATURES = {
            "TransactionAmt", "TransactionDT", "dist1",
            "C1", "C2", "C3", "C5", "C6", "C9", "C11", "C13",
            "V1", "V2", "V3", "V4", "V5",
            "V6", "V7", "V8", "V9", "V10"
    };

    private static final String[] PRODUCTS = {
            "W", "C", "H", "S", "R"
    };

    private static final String[] CATEGORICAL_FEATURES = {
            "P_emaildomain", "R_emaildomain", "card4", "card6", "M4"
    };

    private static final String[] FEATURE_NAMES = createFeatureNames();
    private static final int FEATURE_COUNT = FEATURE_NAMES.length;

    private static final int MAX_ROUNDS = 500;
    private static final int EARLY_STOPPING_ROUNDS = 35;
    private static final int RANDOM_SEED = 42;

    private static final double MIN_PRECISION = 0.80;
    private static final int MIN_ALERTS = 50;

    public static void main(String[] args) throws Exception {

        if (!Files.isRegularFile(CSV_PATH)) {
            throw new IOException("Dataset not found: " + CSV_PATH.toAbsolutePath());
        }

        System.out.println("=== FINFORGE XGBOOST (SUPERVISOR EXPORT) ===");
        System.out.println("Loading chronological splits and encoding categories...");

        DataSplits data = loadSplits();
        Booster model = null;

        try {
            Map<String, Object> params = new LinkedHashMap<>();
            params.put("objective", "binary:logistic");
            params.put("eval_metric", "aucpr");
            params.put("maximize_evaluation_metrics", "true");
            params.put("tree_method", "hist");
            params.put("eta", 0.05);
            params.put("max_depth", 6);
            params.put("min_child_weight", 5);
            params.put("subsample", 0.8);
            params.put("colsample_bytree", 0.8);
            params.put("lambda", 1.0);

            // REVERTED to 1.0 based on Phase C ablation results
            params.put("scale_pos_weight", 1.0);

            params.put("seed", RANDOM_SEED);
            params.put("nthread", 4);
            params.put("verbosity", 0);
            params.put("verbose_eval", 25);

            Map<String, DMatrix> watches = new LinkedHashMap<>();
            watches.put("validation", data.validation);

            float[][] history = new float[1][MAX_ROUNDS];

            System.out.println("\nTraining XGBoost...");
            long start = System.nanoTime();

            model = XGBoost.train(
                    data.train, params, MAX_ROUNDS, watches,
                    history, null, null, EARLY_STOPPING_ROUNDS
            );

            double trainingSeconds = (System.nanoTime() - start) / 1_000_000_000.0;

            String bestIterationText = model.getAttr("best_iteration");
            int bestIteration = bestIterationText == null ? MAX_ROUNDS - 1 : Integer.parseInt(bestIterationText);
            int bestRoundCount = bestIteration + 1;
            String bestScore = model.getAttr("best_score");

            System.out.println("\nTraining complete.");

            float[] validationProbabilities = predictProbabilities(model, data.validation, bestRoundCount);
            float[] testProbabilities = predictProbabilities(model, data.test, bestRoundCount);

            int[] validationLabels = toIntLabels(data.validation.getLabel());
            int[] testLabels = toIntLabels(data.test.getLabel());

            ThresholdChoice choice = chooseThreshold(validationProbabilities, validationLabels);
            Metrics testMetrics = evaluateAtThreshold(testProbabilities, testLabels, choice.threshold);
            RankingMetrics rankings = calculateRankingMetrics(testProbabilities, testLabels);

            double prevalence = (double) countPositive(testLabels) / testLabels.length;

            String report = buildReport(
                    choice, testMetrics, rankings, prevalence,
                    bestRoundCount, bestScore, trainingSeconds
            );

            Path reportDir = Path.of("reports");
            Files.createDirectories(reportDir);

            // Export both Validation (for Supervisor Training) and Test (for Supervisor Evaluation)
            writePredictions(reportDir.resolve("xgboost_validation_predictions.csv"),
                    data.validationMetadata, validationLabels, validationProbabilities, choice.threshold);

            writePredictions(reportDir.resolve("xgboost_test_predictions.csv"),
                    data.testMetadata, testLabels, testProbabilities, choice.threshold);

            System.out.println("\n" + report);
            System.out.println("Exported validation and test predictions for Supervisor Layer.");

        } finally {
            if (model != null) model.dispose();
            data.dispose();
        }
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
        try {
            double value = Double.parseDouble(raw);
            if (!Double.isFinite(value) || value > Float.MAX_VALUE || value < -Float.MAX_VALUE) return Float.NaN;
            return (float) value;
        } catch (NumberFormatException e) {
            return Float.NaN;
        }
    }

    private static DataSplits loadSplits() throws Exception {
        RawSplit trainRaw = new RawSplit(TRAIN_ROWS);
        RawSplit validationRaw = new RawSplit(VALIDATION_ROWS);
        RawSplit testRaw = new RawSplit(TEST_ROWS);

        Map<String, Map<String, Float>> categoryEncoders = new HashMap<>();
        for (String cat : CATEGORICAL_FEATURES) categoryEncoders.put(cat, new HashMap<>());

        long rowIndex = 0;

        try (
                Reader reader = Files.newBufferedReader(CSV_PATH, StandardCharsets.UTF_8);
                CSVParser parser = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).get().parse(reader)
        ) {
            for (CSVRecord row : parser) {
                if (rowIndex >= TOTAL_ROWS) break;

                if (rowIndex < TRAIN_ROWS) {
                    trainRaw.fill(rowIndex, row, categoryEncoders, true);
                } else if (rowIndex < TRAIN_ROWS + VALIDATION_ROWS) {
                    long localIndex = rowIndex - TRAIN_ROWS;
                    validationRaw.fill(localIndex, row, categoryEncoders, false);
                    validationRaw.addMetadata(row); // NOW COLLECTING FOR VALIDATION
                } else {
                    long localIndex = rowIndex - TRAIN_ROWS - VALIDATION_ROWS;
                    testRaw.fill(localIndex, row, categoryEncoders, false);
                    testRaw.addMetadata(row);
                }
                rowIndex++;
            }
        }

        DMatrix trainMatrix = trainRaw.toDMatrix(); trainRaw.releaseArrays();
        DMatrix validationMatrix = validationRaw.toDMatrix(); validationRaw.releaseArrays();
        DMatrix testMatrix = testRaw.toDMatrix(); testRaw.releaseArrays();

        return new DataSplits(trainMatrix, validationMatrix, testMatrix, validationRaw.metadata, testRaw.metadata);
    }

    private static class RawSplit {
        final int rows;
        float[] features;
        float[] labels;
        final List<TestMetadata> metadata = new ArrayList<>();

        RawSplit(int rows) {
            this.rows = rows;
            this.features = new float[rows * FEATURE_COUNT];
            this.labels = new float[rows];
        }

        void fill(long localRow, CSVRecord row, Map<String, Map<String, Float>> encoders, boolean isTraining) {
            int offset = Math.toIntExact(localRow) * FEATURE_COUNT;
            int column = 0;

            for (String feature : NUMERIC_FEATURES) features[offset + column++] = parseNumber(row.get(feature));
            String product = row.get("ProductCD");
            for (String category : PRODUCTS) features[offset + column++] = category.equals(product) ? 1.0f : 0.0f;

            for (String cat : CATEGORICAL_FEATURES) {
                String val = row.get(cat);
                if (val == null || val.isBlank()) {
                    features[offset + column++] = Float.NaN;
                } else {
                    Map<String, Float> dict = encoders.get(cat);
                    if (isTraining) {
                        float id = dict.computeIfAbsent(val, k -> (float) dict.size() + 1f);
                        features[offset + column++] = id;
                    } else {
                        Float id = dict.get(val);
                        features[offset + column++] = (id != null) ? id : Float.NaN;
                    }
                }
            }
            labels[Math.toIntExact(localRow)] = Float.parseFloat(row.get("isFraud"));
        }

        void addMetadata(CSVRecord row) {
            metadata.add(new TestMetadata(row.get("TransactionID"), row.get("TransactionDT"), row.get("TransactionAmt"), row.get("ProductCD")));
        }

        DMatrix toDMatrix() throws Exception {
            DMatrix matrix = new DMatrix(features, rows, FEATURE_COUNT, Float.NaN);
            matrix.setLabel(labels);
            matrix.setFeatureNames(FEATURE_NAMES);
            return matrix;
        }

        void releaseArrays() { features = null; labels = null; }
    }

    private static class DataSplits {
        final DMatrix train, validation, test;
        final List<TestMetadata> validationMetadata;
        final List<TestMetadata> testMetadata;

        DataSplits(DMatrix train, DMatrix validation, DMatrix test, List<TestMetadata> validationMetadata, List<TestMetadata> testMetadata) {
            this.train = train; this.validation = validation; this.test = test;
            this.validationMetadata = validationMetadata; this.testMetadata = testMetadata;
        }
        void dispose() { train.dispose(); validation.dispose(); test.dispose(); }
    }

    private static class TestMetadata {
        final String transactionId, transactionDT, amount, product;
        TestMetadata(String transactionId, String transactionDT, String amount, String product) {
            this.transactionId = transactionId; this.transactionDT = transactionDT;
            this.amount = amount; this.product = product;
        }
    }

    private static float[] predictProbabilities(Booster model, DMatrix matrix, int bestRoundCount) throws Exception {
        float[][] raw = model.predict(matrix, false, bestRoundCount);
        float[] probs = new float[raw.length];
        for (int i = 0; i < raw.length; i++) probs[i] = raw[i][0];
        return probs;
    }

    private static int[] toIntLabels(float[] rawLabels) {
        int[] labels = new int[rawLabels.length];
        for (int i = 0; i < rawLabels.length; i++) labels[i] = Math.round(rawLabels[i]);
        return labels;
    }

    private static ThresholdChoice chooseThreshold(float[] probabilities, int[] labels) {
        ThresholdChoice bestFeasible = null;
        for (int step = 1; step <= 99; step++) {
            double threshold = step / 100.0;
            Metrics candidate = evaluateAtThreshold(probabilities, labels, threshold);

            if (candidate.precision() < MIN_PRECISION || candidate.alerts() < MIN_ALERTS) continue;

            if (bestFeasible == null || candidate.recall() > bestFeasible.metrics.recall()
                    || (candidate.recall() == bestFeasible.metrics.recall() && candidate.precision() > bestFeasible.metrics.precision())) {
                bestFeasible = new ThresholdChoice(threshold, candidate, true);
            }
        }
        if (bestFeasible != null) return bestFeasible;

        ThresholdChoice bestF1 = null;
        for (int step = 1; step <= 99; step++) {
            double threshold = step / 100.0;
            Metrics candidate = evaluateAtThreshold(probabilities, labels, threshold);
            if (bestF1 == null || candidate.f1() > bestF1.metrics.f1()) bestF1 = new ThresholdChoice(threshold, candidate, false);
        }
        return bestF1;
    }

    private static Metrics evaluateAtThreshold(float[] probabilities, int[] labels, double threshold) {
        Metrics result = new Metrics();
        for (int i = 0; i < labels.length; i++) {
            boolean predicted = probabilities[i] >= threshold;
            boolean actual = labels[i] == 1;

            if (predicted && actual) result.tp++;
            else if (predicted) result.fp++;
            else if (actual) result.fn++;
            else result.tn++;
        }
        return result;
    }

    private static RankingMetrics calculateRankingMetrics(float[] probabilities, int[] labels) {
        ScoreLabel[] sorted = new ScoreLabel[probabilities.length];
        long positives = 0;
        for (int i = 0; i < probabilities.length; i++) {
            sorted[i] = new ScoreLabel(probabilities[i], labels[i]);
            if (labels[i] == 1) positives++;
        }
        long negatives = labels.length - positives;
        if (positives == 0 || negatives == 0) return new RankingMetrics(Double.NaN, Double.NaN);

        Arrays.sort(sorted, Comparator.comparingDouble((ScoreLabel x) -> x.probability).reversed());
        long tp = 0, fp = 0;
        double prevRecall = 0, prevFpr = 0, prevPrecision = 1, rocAuc = 0, prAuc = 0;
        int i = 0;

        while (i < sorted.length) {
            float currentScore = sorted[i].probability;
            int j = i;
            while (j < sorted.length && Float.compare(sorted[j].probability, currentScore) == 0) {
                if (sorted[j].label == 1) tp++;
                else fp++;
                j++;
            }
            double recall = (double) tp / positives;
            double fpr = (double) fp / negatives;
            double precision = tp + fp == 0 ? 1 : (double) tp / (tp + fp);

            rocAuc += (fpr - prevFpr) * (recall + prevRecall) / 2.0;
            prAuc += (recall - prevRecall) * (precision + prevPrecision) / 2.0;

            prevRecall = recall; prevFpr = fpr; prevPrecision = precision;
            i = j;
        }
        return new RankingMetrics(rocAuc, prAuc);
    }

    private static long countPositive(int[] labels) {
        long total = 0;
        for (int label : labels) if (label == 1) total++;
        return total;
    }

    private static class ScoreLabel {
        final float probability; final int label;
        ScoreLabel(float probability, int label) { this.probability = probability; this.label = label; }
    }

    private static class Metrics {
        long tp, fp, fn, tn;
        long alerts() { return tp + fp; }
        double precision() { return tp + fp == 0 ? 0 : (double) tp / (tp + fp); }
        double recall() { return tp + fn == 0 ? 0 : (double) tp / (tp + fn); }
        double specificity() { return tn + fp == 0 ? 0 : (double) tn / (tn + fp); }
        double f1() { double p = precision(), r = recall(); return p + r == 0 ? 0 : 2 * p * r / (p + r); }
    }

    private static class RankingMetrics {
        final double rocAuc, prAuc;
        RankingMetrics(double rocAuc, double prAuc) { this.rocAuc = rocAuc; this.prAuc = prAuc; }
    }

    private static class ThresholdChoice {
        final double threshold; final Metrics metrics; final boolean targetMet;
        ThresholdChoice(double threshold, Metrics metrics, boolean targetMet) {
            this.threshold = threshold; this.metrics = metrics; this.targetMet = targetMet;
        }
    }

    private static String buildReport(ThresholdChoice choice, Metrics test, RankingMetrics ranking, double prevalence, int bestRoundCount, String bestScore, double trainingSeconds) {
        StringBuilder out = new StringBuilder();
        out.append("=== FINFORGE XGBOOST BENCHMARK ===\n\nDATA SPLIT\n--------------------------------------\n");
        out.append("Training rows:        ").append(TRAIN_ROWS).append("\nValidation rows:      ").append(VALIDATION_ROWS).append("\nTest rows:            ").append(TEST_ROWS).append("\n");
        return out.toString();
    }

    private static void writePredictions(Path path, List<TestMetadata> metadata, int[] labels, float[] probabilities, double threshold) throws IOException {
        try (var writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8);
             CSVPrinter printer = new CSVPrinter(writer, CSVFormat.DEFAULT)) {
            printer.printRecord("TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", "actualFraud", "fraudProbability", "predictedFraud", "threshold");
            for (int i = 0; i < probabilities.length; i++) {
                printer.printRecord(metadata.get(i).transactionId, metadata.get(i).transactionDT, metadata.get(i).amount, metadata.get(i).product,
                        labels[i], String.format(Locale.US, "%.8f", probabilities[i]), (probabilities[i] >= threshold ? 1 : 0), String.format(Locale.US, "%.4f", threshold));
            }
        }
    }
}