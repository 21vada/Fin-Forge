package com.finforge;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import weka.classifiers.functions.Logistic;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

public class FraudSupervisor {

    private static final Path REPORTS_DIR = Path.of("reports");

    // Training on out-of-sample validation outputs prevents data leakage
    private static final Path RF_VAL_PRED = REPORTS_DIR.resolve("random_forest_validation_predictions.csv");
    private static final Path XGB_VAL_PRED = REPORTS_DIR.resolve("xgboost_validation_predictions.csv");

    // Testing strictly on held-out test predictions
    private static final Path RF_TEST_PRED = REPORTS_DIR.resolve("random_forest_test_predictions.csv");
    private static final Path XGB_TEST_PRED = REPORTS_DIR.resolve("xgboost_test_predictions.csv");

    private static final double MIN_PRECISION = 0.80;

    public static void main(String[] args) throws Exception {

        System.out.println("=== FINFORGE LAYER 2 SUPERVISOR ===");

        System.out.println("Loading and aligning Validation predictions (Supervisor Training Data)...");
        Instances supervisorTrain = loadMetaDataset("SupervisorTrain", RF_VAL_PRED, XGB_VAL_PRED);

        System.out.println("Loading and aligning Test predictions (Supervisor Evaluation Data)...");
        Instances supervisorTest = loadMetaDataset("SupervisorTest", RF_TEST_PRED, XGB_TEST_PRED);

        System.out.println("\nTraining Logistic Regression Meta-Model...");
        Logistic metaModel = new Logistic();
        metaModel.buildClassifier(supervisorTrain);

        System.out.println("Selecting operating threshold...");
        double[] valProbs = predictProbabilities(metaModel, supervisorTrain);
        int[] valLabels = actualLabels(supervisorTrain);

        double bestThreshold = chooseThreshold(valProbs, valLabels);
        System.out.printf(Locale.US, "Selected Threshold: %.2f%n", bestThreshold);

        System.out.println("\nEvaluating Meta-Model on Held-Out Test Set...");
        double[] testProbs = predictProbabilities(metaModel, supervisorTest);
        int[] testLabels = actualLabels(supervisorTest);

        Metrics testMetrics = evaluateAtThreshold(testProbs, testLabels, bestThreshold);
        RankingMetrics ranking = calculateRankingMetrics(testProbs, testLabels);

        printReport(testMetrics, ranking);
    }

    private static Instances loadMetaDataset(String name, Path rfPath, Path xgbPath) throws IOException {

        Map<String, BasePrediction> rfData = parsePredictions(rfPath);
        Map<String, BasePrediction> xgbData = parsePredictions(xgbPath);

        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("rfProbability"));
        attributes.add(new Attribute("xgbProbability"));
        attributes.add(new Attribute("isFraud", new ArrayList<>(Arrays.asList("0", "1"))));

        Instances dataset = new Instances(name, attributes, rfData.size());
        dataset.setClassIndex(2);

        for (Map.Entry<String, BasePrediction> entry : xgbData.entrySet()) {
            String txId = entry.getKey();
            BasePrediction xgb = entry.getValue();
            BasePrediction rf = rfData.get(txId);

            if (rf == null || rf.actualFraud != xgb.actualFraud) {
                throw new IllegalStateException("Misaligned transaction or labels: " + txId);
            }

            double[] values = new double[]{ rf.probability, xgb.probability, xgb.actualFraud };
            dataset.add(new DenseInstance(1.0, values));
        }

        return dataset;
    }

    private static Map<String, BasePrediction> parsePredictions(Path path) throws IOException {
        Map<String, BasePrediction> map = new HashMap<>();
        try (
                Reader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8);
                CSVParser parser = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).get().parse(reader)
        ) {
            for (CSVRecord row : parser) {
                map.put(row.get("TransactionID"), new BasePrediction(
                        Integer.parseInt(row.get("actualFraud")),
                        Double.parseDouble(row.get("fraudProbability"))
                ));
            }
        }
        return map;
    }

    private static class BasePrediction {
        final int actualFraud;
        final double probability;
        BasePrediction(int actualFraud, double probability) {
            this.actualFraud = actualFraud;
            this.probability = probability;
        }
    }

    private static double[] predictProbabilities(Logistic model, Instances dataset) throws Exception {
        int posIndex = dataset.classAttribute().indexOfValue("1");
        double[] probs = new double[dataset.numInstances()];
        for (int i = 0; i < dataset.numInstances(); i++) {
            probs[i] = model.distributionForInstance(dataset.instance(i))[posIndex];
        }
        return probs;
    }

    private static int[] actualLabels(Instances dataset) {
        int[] labels = new int[dataset.numInstances()];
        for (int i = 0; i < dataset.numInstances(); i++) {
            labels[i] = (int) dataset.instance(i).classValue();
        }
        return labels;
    }

    private static double chooseThreshold(double[] probabilities, int[] labels) {
        double bestThreshold = 0.50;
        double bestRecall = -1;

        for (int step = 1; step <= 99; step++) {
            double threshold = step / 100.0;
            Metrics m = evaluateAtThreshold(probabilities, labels, threshold);

            if (m.precision() >= MIN_PRECISION && m.alerts() >= 50) {
                if (m.recall() > bestRecall) {
                    bestRecall = m.recall();
                    bestThreshold = threshold;
                }
            }
        }
        return bestThreshold;
    }

    private static Metrics evaluateAtThreshold(double[] probabilities, int[] labels, double threshold) {
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

    private static RankingMetrics calculateRankingMetrics(double[] probabilities, int[] labels) {
        ScoreLabel[] sorted = new ScoreLabel[probabilities.length];
        long positives = 0;
        for (int i = 0; i < probabilities.length; i++) {
            sorted[i] = new ScoreLabel(probabilities[i], labels[i]);
            if (labels[i] == 1) positives++;
        }
        long negatives = labels.length - positives;

        Arrays.sort(sorted, Comparator.comparingDouble((ScoreLabel x) -> x.probability).reversed());
        long tp = 0, fp = 0;
        double prevRecall = 0, prevFpr = 0, prevPrecision = 1, rocAuc = 0, prAuc = 0;
        int i = 0;

        while (i < sorted.length) {
            double currentScore = sorted[i].probability;
            int j = i;
            while (j < sorted.length && Double.compare(sorted[j].probability, currentScore) == 0) {
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

    private static class ScoreLabel {
        final double probability; final int label;
        ScoreLabel(double probability, int label) { this.probability = probability; this.label = label; }
    }

    private static class Metrics {
        long tp, fp, fn, tn;
        long alerts() { return tp + fp; }
        double precision() { return tp + fp == 0 ? 0 : (double) tp / (tp + fp); }
        double recall() { return tp + fn == 0 ? 0 : (double) tp / (tp + fn); }
        double f1() { double p = precision(), r = recall(); return p + r == 0 ? 0 : 2 * p * r / (p + r); }
    }

    private static class RankingMetrics {
        final double rocAuc, prAuc;
        RankingMetrics(double rocAuc, double prAuc) { this.rocAuc = rocAuc; this.prAuc = prAuc; }
    }

    private static void printReport(Metrics test, RankingMetrics ranking) {
        System.out.println("\n--------------------------------------");
        System.out.println("Alerts:          " + test.alerts());
        System.out.println("True positives:  " + test.tp);
        System.out.println("False positives: " + test.fp);
        System.out.println("False negatives: " + test.fn);
        System.out.println("True negatives:  " + test.tn);
        System.out.printf(Locale.US, "Precision:       %.2f%%%n", test.precision() * 100);
        System.out.printf(Locale.US, "Recall:          %.2f%%%n", test.recall() * 100);
        System.out.printf(Locale.US, "F1 score:        %.2f%%%n", test.f1() * 100);
        System.out.println("--------------------------------------");
        System.out.printf(Locale.US, "ROC-AUC:         %.4f%n", ranking.rocAuc);
        System.out.printf(Locale.US, "PR-AUC:          %.4f%n", ranking.prAuc);
    }
}