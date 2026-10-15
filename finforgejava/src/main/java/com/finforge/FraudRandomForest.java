package com.finforge;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

public class FraudRandomForest {

    private static final Path CSV_PATH = Path.of(
            "C:/Users/pc/Documents/fin forge/ieee-fraud-detection/train_transaction.csv"
    );

    private static final int TRAIN_ROWS = 413_378;
    private static final int VALIDATION_ROWS = 88_581;
    private static final int TEST_ROWS = 88_581;
    private static final int TOTAL_ROWS = TRAIN_ROWS + VALIDATION_ROWS + TEST_ROWS;

    private static final String[] NUMERIC_FEATURES = {
            "TransactionAmt", "TransactionDT", "dist1",
            "C1", "C2", "C3", "C5", "C6", "C9", "C11", "C13",
            "V1", "V2", "V3", "V4", "V5",
            "V6", "V7", "V8", "V9", "V10"
    };

    private static final String[] PRODUCTS = {
            "W", "C", "H", "S", "R"
    };

    private static final int TREE_COUNT = 100;
    private static final int MAX_DEPTH = 18;
    private static final int FEATURES_PER_SPLIT = 5;
    private static final int RANDOM_SEED = 42;

    private static final double MIN_PRECISION = 0.80;
    private static final int MIN_VALIDATION_ALERTS = 50;

    public static void main(String[] args) throws Exception {

        if (!Files.isRegularFile(CSV_PATH)) {
            throw new IOException("Dataset not found: " + CSV_PATH.toAbsolutePath());
        }

        System.out.println("=== FINFORGE RANDOM FOREST (SUPERVISOR EXPORT) ===");
        System.out.println("Loading chronological dataset splits...");

        DataSplits data = loadSplits();

        double[] trainingMeans = calculateTrainingMeans(data.train);
        applyTrainingMeans(data.train, trainingMeans);
        applyTrainingMeans(data.validation, trainingMeans);
        applyTrainingMeans(data.test, trainingMeans);

        System.out.println("\nTraining Random Forest...");

        RandomForest model = new RandomForest();
        model.setNumIterations(TREE_COUNT);
        model.setSeed(RANDOM_SEED);
        model.setMaxDepth(MAX_DEPTH);
        model.setNumFeatures(FEATURES_PER_SPLIT);

        long startTime = System.nanoTime();
        model.buildClassifier(data.train);
        double trainingSeconds = (System.nanoTime() - startTime) / 1_000_000_000.0;

        System.out.println("Training complete.");

        double[] validationProbabilities = positiveProbabilities(model, data.validation);
        int[] validationLabels = actualLabels(data.validation);
        ThresholdChoice choice = chooseThreshold(validationProbabilities, validationLabels);

        double[] testProbabilities = positiveProbabilities(model, data.test);
        int[] testLabels = actualLabels(data.test);
        Metrics testMetrics = calculateMetrics(testProbabilities, testLabels, choice.threshold);

        Evaluation evaluation = new Evaluation(data.train);
        evaluation.evaluateModel(model, data.test);
        int fraudClassIndex = data.test.classAttribute().indexOfValue("1");

        double rocAuc = evaluation.areaUnderROC(fraudClassIndex);
        double prAuc = evaluation.areaUnderPRC(fraudClassIndex);

        String report = buildReport(data, choice, testMetrics, rocAuc, prAuc, trainingSeconds);

        Path reportsDirectory = Path.of("reports");
        Files.createDirectories(reportsDirectory);

        // Export validation and test probabilities for the Supervisor layer
        writePredictions(reportsDirectory.resolve("random_forest_validation_predictions.csv"),
                data.validationMetadata, validationProbabilities, choice.threshold);

        writePredictions(reportsDirectory.resolve("random_forest_test_predictions.csv"),
                data.testMetadata, testProbabilities, choice.threshold);

        Files.writeString(reportsDirectory.resolve("random_forest_temporal_benchmark.txt"), report, StandardCharsets.UTF_8);

        System.out.println("\n" + report);
        System.out.println("Exported validation and test predictions for Supervisor Layer.");
    }

    private static DataSplits loadSplits() throws IOException {
        Instances train = createSchema("FinForgeTrain", TRAIN_ROWS);
        Instances validation = createSchema("FinForgeValidation", VALIDATION_ROWS);
        Instances test = createSchema("FinForgeTest", TEST_ROWS);

        List<TestMetadata> validationMetadata = new ArrayList<>(VALIDATION_ROWS);
        List<TestMetadata> testMetadata = new ArrayList<>(TEST_ROWS);

        long rowIndex = 0;
        try (
                Reader reader = Files.newBufferedReader(CSV_PATH, StandardCharsets.UTF_8);
                CSVParser parser = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).get().parse(reader)
        ) {
            for (CSVRecord row : parser) {
                if (rowIndex >= TOTAL_ROWS) break;

                double[] values = makeFeatureVector(row);

                if (rowIndex < TRAIN_ROWS) {
                    train.add(new DenseInstance(1.0, values));
                } else if (rowIndex < TRAIN_ROWS + VALIDATION_ROWS) {
                    validation.add(new DenseInstance(1.0, values));
                    validationMetadata.add(new TestMetadata(row.get("TransactionID"), row.get("TransactionDT"), row.get("TransactionAmt"), row.get("ProductCD"), Integer.parseInt(row.get("isFraud"))));
                } else {
                    test.add(new DenseInstance(1.0, values));
                    testMetadata.add(new TestMetadata(row.get("TransactionID"), row.get("TransactionDT"), row.get("TransactionAmt"), row.get("ProductCD"), Integer.parseInt(row.get("isFraud"))));
                }
                rowIndex++;
            }
        }
        return new DataSplits(train, validation, test, validationMetadata, testMetadata);
    }

    private static Instances createSchema(String relationName, int capacity) {
        ArrayList<Attribute> attributes = new ArrayList<>();
        for (String feature : NUMERIC_FEATURES) attributes.add(new Attribute(feature));
        for (String product : PRODUCTS) attributes.add(new Attribute("ProductCD_" + product));
        attributes.add(new Attribute("isFraud", new ArrayList<>(Arrays.asList("0", "1"))));
        Instances dataset = new Instances(relationName, attributes, capacity);
        dataset.setClassIndex(dataset.numAttributes() - 1);
        return dataset;
    }

    private static double[] makeFeatureVector(CSVRecord row) {
        double[] values = new double[NUMERIC_FEATURES.length + PRODUCTS.length + 1];
        int index = 0;
        for (String feature : NUMERIC_FEATURES) values[index++] = parseNumber(row.get(feature));
        String product = row.get("ProductCD");
        for (String category : PRODUCTS) values[index++] = category.equals(product) ? 1.0 : 0.0;
        values[index] = Integer.parseInt(row.get("isFraud"));
        return values;
    }

    private static double parseNumber(String raw) {
        if (raw == null || raw.isBlank()) return Double.NaN;
        try {
            double value = Double.parseDouble(raw);
            return Double.isFinite(value) ? value : Double.NaN;
        } catch (NumberFormatException ex) {
            return Double.NaN;
        }
    }

    private static double[] calculateTrainingMeans(Instances training) {
        int featureCount = training.numAttributes() - 1;
        double[] sums = new double[featureCount];
        long[] counts = new long[featureCount];

        for (Instance row : training) {
            for (int feature = 0; feature < featureCount; feature++) {
                if (!row.isMissing(feature)) {
                    sums[feature] += row.value(feature);
                    counts[feature]++;
                }
            }
        }
        double[] means = new double[featureCount];
        for (int feature = 0; feature < featureCount; feature++) {
            means[feature] = counts[feature] == 0 ? 0.0 : sums[feature] / counts[feature];
        }
        return means;
    }

    private static void applyTrainingMeans(Instances dataset, double[] trainingMeans) {
        for (Instance row : dataset) {
            for (int feature = 0; feature < trainingMeans.length; feature++) {
                if (row.isMissing(feature)) row.setValue(feature, trainingMeans[feature]);
            }
        }
    }

    private static ThresholdChoice chooseThreshold(double[] probabilities, int[] labels) {
        double bestThreshold = Double.NaN;
        Metrics bestMetrics = null;

        for (int step = 1; step <= 99; step++) {
            double threshold = step / 100.0;
            Metrics candidate = calculateMetrics(probabilities, labels, threshold);

            if (candidate.alerts() < MIN_VALIDATION_ALERTS || candidate.precision() < MIN_PRECISION) continue;

            if (bestMetrics == null || candidate.recall() > bestMetrics.recall() ||
                    (Math.abs(candidate.recall() - bestMetrics.recall()) < 1e-12 && candidate.precision() > bestMetrics.precision())) {
                bestThreshold = threshold;
                bestMetrics = candidate;
            }
        }
        if (bestMetrics != null) return new ThresholdChoice(bestThreshold, bestMetrics, true);

        for (int step = 1; step <= 99; step++) {
            double threshold = step / 100.0;
            Metrics candidate = calculateMetrics(probabilities, labels, threshold);
            if (bestMetrics == null || candidate.f1() > bestMetrics.f1()) {
                bestThreshold = threshold;
                bestMetrics = candidate;
            }
        }
        return new ThresholdChoice(bestThreshold, bestMetrics, false);
    }

    private static double[] positiveProbabilities(RandomForest model, Instances dataset) throws Exception {
        int posIndex = dataset.classAttribute().indexOfValue("1");
        double[] probs = new double[dataset.numInstances()];
        for (int i = 0; i < dataset.numInstances(); i++) probs[i] = model.distributionForInstance(dataset.instance(i))[posIndex];
        return probs;
    }

    private static int[] actualLabels(Instances dataset) {
        int[] labels = new int[dataset.numInstances()];
        for (int i = 0; i < dataset.numInstances(); i++) labels[i] = (int) dataset.instance(i).classValue();
        return labels;
    }

    private static Metrics calculateMetrics(double[] probabilities, int[] labels, double threshold) {
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

    private static long countFraud(Instances dataset) {
        long fraud = 0;
        for (Instance row : dataset) if ((int) row.classValue() == 1) fraud++;
        return fraud;
    }

    private static String buildReport(DataSplits data, ThresholdChoice choice, Metrics test, double rocAuc, double prAuc, double trainingSeconds) {
        StringBuilder out = new StringBuilder();
        out.append("=== FINFORGE RANDOM FOREST BENCHMARK ===\n\n");
        out.append(String.format(Locale.US, "Training time: %.1f seconds%n", trainingSeconds));
        out.append(String.format(Locale.US, "Selected threshold: %.2f%n", choice.threshold));
        out.append("Validation precision target met: ").append(choice.targetMet).append('\n');
        appendMetrics(out, "HELD-OUT TEST METRICS", test);
        out.append("\nRANKING METRICS\n--------------------------------------\n");
        out.append(String.format(Locale.US, "ROC-AUC:               %.4f%n", rocAuc));
        out.append(String.format(Locale.US, "PR-AUC:                %.4f%n", prAuc));
        return out.toString();
    }

    private static void appendMetrics(StringBuilder out, String title, Metrics metrics) {
        out.append('\n').append(title).append("\n--------------------------------------\n");
        out.append("Alerts:          ").append(metrics.alerts()).append('\n');
        out.append("True positives:  ").append(metrics.tp).append('\n');
        out.append("False positives: ").append(metrics.fp).append('\n');
        out.append("False negatives: ").append(metrics.fn).append('\n');
        out.append("True negatives:  ").append(metrics.tn).append('\n');
        out.append("Precision:       ").append(percent(metrics.tp, metrics.tp + metrics.fp)).append('\n');
        out.append("Recall:          ").append(percent(metrics.tp, metrics.tp + metrics.fn)).append('\n');
        out.append("F1 score:        ").append(percent(metrics.f1())).append('\n');
    }

    private static String percent(long num, long den) { return den == 0 ? "N/A" : String.format(Locale.US, "%.2f%%", 100.0 * num / den); }
    private static String percent(double val) { return String.format(Locale.US, "%.2f%%", val * 100.0); }

    private static void writePredictions(Path outputPath, List<TestMetadata> metadata, double[] probabilities, double threshold) throws IOException {
        try (var writer = Files.newBufferedWriter(outputPath, StandardCharsets.UTF_8); CSVPrinter printer = new CSVPrinter(writer, CSVFormat.DEFAULT)) {
            printer.printRecord("TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", "actualFraud", "fraudProbability", "predictedFraud", "threshold");
            for (int i = 0; i < probabilities.length; i++) {
                TestMetadata row = metadata.get(i);
                printer.printRecord(row.transactionId, row.transactionDT, row.amount, row.product, row.actualFraud, String.format(Locale.US, "%.8f", probabilities[i]), (probabilities[i] >= threshold ? 1 : 0), String.format(Locale.US, "%.4f", threshold));
            }
        }
    }

    private static class DataSplits {
        final Instances train, validation, test;
        final List<TestMetadata> validationMetadata, testMetadata;
        DataSplits(Instances train, Instances validation, Instances test, List<TestMetadata> validationMetadata, List<TestMetadata> testMetadata) {
            this.train = train; this.validation = validation; this.test = test;
            this.validationMetadata = validationMetadata; this.testMetadata = testMetadata;
        }
    }

    private static class TestMetadata {
        final String transactionId, transactionDT, amount, product;
        final int actualFraud;
        TestMetadata(String transactionId, String transactionDT, String amount, String product, int actualFraud) {
            this.transactionId = transactionId; this.transactionDT = transactionDT; this.amount = amount; this.product = product; this.actualFraud = actualFraud;
        }
    }

    private static class Metrics {
        long tp, fp, fn, tn;
        long alerts() { return tp + fp; }
        double precision() { return tp + fp == 0 ? 0.0 : (double) tp / (tp + fp); }
        double recall() { return tp + fn == 0 ? 0.0 : (double) tp / (tp + fn); }
        double f1() { double p = precision(), r = recall(); return p + r == 0 ? 0.0 : 2.0 * p * r / (p + r); }
    }

    private static class ThresholdChoice {
        final double threshold; final Metrics metrics; final boolean targetMet;
        ThresholdChoice(double threshold, Metrics metrics, boolean targetMet) {
            this.threshold = threshold; this.metrics = metrics; this.targetMet = targetMet;
        }
    }
}