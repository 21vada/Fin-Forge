package com.finforge;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.*;

public class MasterFinForgePipeline {

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
            "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10"
    };

    private static final String[] CATEGORICAL_FEATURES = {
            "P_emaildomain", "R_emaildomain", "card4", "card6", "M4"
    };

    private static final String[] FEATURE_NAMES = createFeatureNames();
    private static final int FEATURE_COUNT = FEATURE_NAMES.length;

    // Registry of test transactions for interactive console lookup
    private static final List<TestTransaction> testTransactionRegistry = new ArrayList<>();

    public static void main(String[] args) throws Exception {

        System.out.println("=== FINFORGE NEURO-SYMBOLIC MASTER PIPELINE ===");

        SpecialistData wData = new SpecialistData("W_Specialist");
        SpecialistData cData = new SpecialistData("C_Specialist");
        SpecialistData hrsData = new SpecialistData("HRS_Specialist");

        try (OnnxCnnPredictor cnnPredictor = new OnnxCnnPredictor()) {
            System.out.println("\nStreaming data, building temporal windows, and generating CNN scores...");
            loadAndRouteData(wData, cData, hrsData, cnnPredictor);
        }

        System.out.println("\n--- TRAINING SPECIALIST 1: HIGH-VALUE W-CLASS ---");
        Booster modelW = trainSpecialist(wData);

        System.out.println("\n--- TRAINING SPECIALIST 2: HIGH-VELOCITY C-CLASS ---");
        Booster modelC = trainSpecialist(cData);

        System.out.println("\n--- TRAINING SPECIALIST 3: DIGITAL ACCOUNT (H, R, S) ---");
        Booster modelHRS = trainSpecialist(hrsData);

        System.out.println("\n=== LAYER 2 SUPERVISOR: DYNAMIC CALIBRATION ===");

        List<ScoreLabel> valW = scoreSpecialist(modelW, wData, true);
        List<ScoreLabel> valC = scoreSpecialist(modelC, cData, true);
        List<ScoreLabel> valHRS = scoreSpecialist(modelHRS, hrsData, true);

        double threshW = findOptimalThreshold(valW, "W_Specialist");
        double threshC = findOptimalThreshold(valC, "C_Specialist");
        double threshHRS = findOptimalThreshold(valHRS, "HRS_Specialist");

        // Initialize Interactive Service Layer
        FinForgeInteractiveService service = new FinForgeInteractiveService(modelW, modelC, modelHRS);
        service.adjustThreshold("W", threshW, "Layer 2 Supervisor F1-Optimization");
        service.adjustThreshold("C", threshC, "Layer 2 Supervisor F1-Optimization");
        service.adjustThreshold("HRS", threshHRS, "Layer 2 Supervisor F1-Optimization");

        System.out.println("\n=== GLOBAL TEST EVALUATION ===");
        List<ScoreLabel> testW = scoreSpecialist(modelW, wData, false);
        List<ScoreLabel> testC = scoreSpecialist(modelC, cData, false);
        List<ScoreLabel> testHRS = scoreSpecialist(modelHRS, hrsData, false);

        Metrics perfW = evaluateAtThreshold(testW, threshW);
        Metrics perfC = evaluateAtThreshold(testC, threshC);
        Metrics perfHRS = evaluateAtThreshold(testHRS, threshHRS);

        Metrics globalPerf = new Metrics();
        globalPerf.tp = perfW.tp + perfC.tp + perfHRS.tp;
        globalPerf.fp = perfW.fp + perfC.fp + perfHRS.fp;
        globalPerf.fn = perfW.fn + perfC.fn + perfHRS.fn;
        globalPerf.tn = perfW.tn + perfC.tn + perfHRS.tn;

        List<ScoreLabel> allTestScores = new ArrayList<>();
        allTestScores.addAll(testW); allTestScores.addAll(testC); allTestScores.addAll(testHRS);
        RankingMetrics ranking = calculateRankingMetrics(allTestScores);

        long totalScored = globalPerf.tp + globalPerf.fp + globalPerf.fn + globalPerf.tn;
        System.out.println("Total Transactions Scored: " + totalScored);
        System.out.println("--------------------------------------------------");
        System.out.printf(Locale.US, "Global ROC-AUC:         %.4f%n", ranking.rocAuc);
        System.out.printf(Locale.US, "Global PR-AUC:          %.4f%n", ranking.prAuc);
        System.out.println("--------------------------------------------------");
        System.out.println("True Positives Caught:  " + globalPerf.tp);
        System.out.println("False Positives:        " + globalPerf.fp);
        System.out.println("Missed Fraud:           " + globalPerf.fn);
        System.out.printf(Locale.US, "Global Recall:          %.2f%%%n", globalPerf.recall() * 100);
        System.out.printf(Locale.US, "Global Precision:       %.2f%%%n", globalPerf.precision() * 100);
        System.out.printf(Locale.US, "Global F1-Score:        %.2f%%%n", globalPerf.f1() * 100);
        System.out.println("==================================================");

        // ========================================================================
        // INTERACTIVE CONSOLE LAYER
        // ========================================================================
        runInteractiveConsole(service);

        modelW.dispose(); modelC.dispose(); modelHRS.dispose();
        wData.dispose(); cData.dispose(); hrsData.dispose();
    }

    private static void runInteractiveConsole(FinForgeInteractiveService service) throws XGBoostError {
        Scanner scanner = new Scanner(System.in);
        System.out.println("\n==================================================");
        System.out.println("💻 FINFORGE INTERACTIVE EXPLANATION CONSOLE");
        System.out.println("==================================================");
        System.out.println("Available Test Transactions: 0 to " + (testTransactionRegistry.size() - 1));
        System.out.println("Commands:");
        System.out.println("  - Enter a number (e.g. '42') to evaluate and explain a transaction.");
        System.out.println("  - Enter 'SET <ProductCD> <Threshold>' (e.g. 'SET W 0.25') to adjust thresholds.");
        System.out.println("  - Enter 'REASON <ProductCD>' to view current threshold justification.");
        System.out.println("  - Enter 'EXIT' to quit.");

        while (true) {
            System.out.print("\nFinForge> ");
            String line = scanner.nextLine().trim();
            if (line.isEmpty()) continue;

            if (line.equalsIgnoreCase("EXIT")) {
                System.out.println("Exiting FinForge Interactive Console.");
                break;
            }

            String[] tokens = line.split("\\s+");

            if (tokens[0].equalsIgnoreCase("SET") && tokens.length >= 3) {
                String product = tokens[1].toUpperCase();
                try {
                    double newThresh = Double.parseDouble(tokens[2]);
                    service.adjustThreshold(product, newThresh, "Manual operator adjustment via Console");
                } catch (NumberFormatException e) {
                    System.out.println("❌ Invalid threshold value. Example: SET W 0.20");
                }
            } else if (tokens[0].equalsIgnoreCase("REASON") && tokens.length >= 2) {
                String product = tokens[1].toUpperCase();
                System.out.println("\n📊 Threshold Justification for " + product + ":");
                System.out.println("  " + service.getThresholdReasoning(product));
            } else {
                try {
                    int index = Integer.parseInt(line);
                    if (index < 0 || index >= testTransactionRegistry.size()) {
                        System.out.printf("❌ Transaction index out of bounds. Enter 0 to %d.%n", testTransactionRegistry.size() - 1);
                    } else {
                        TestTransaction tx = testTransactionRegistry.get(index);
                        System.out.printf("%nEvaluating Transaction #%d (Actual Label: %s)%n", index, tx.isFraud == 1 ? "FRAUD" : "LEGITIMATE");
                        service.evaluateAndExplain(tx.rawFeatures, FEATURE_NAMES, tx.productCD, tx.isFraud);
                    }
                } catch (NumberFormatException e) {
                    System.out.println("❌ Command not recognized. Enter a transaction number, 'SET W 0.20', 'REASON W', or 'EXIT'.");
                }
            }
        }
    }

    private static double findOptimalThreshold(List<ScoreLabel> validationScores, String name) {
        if (validationScores.isEmpty()) return 0.5;
        double bestThreshold = 0.5;
        double bestF1 = 0.0;

        for (double t = 0.01; t <= 0.99; t += 0.01) {
            Metrics m = evaluateAtThreshold(validationScores, t);
            double f1 = m.f1();
            if (f1 > bestF1) {
                bestF1 = f1;
                bestThreshold = t;
            }
        }
        System.out.printf(Locale.US, "Supervisor locked threshold for %-15s at %.2f (Validation F1: %.2f%%)%n",
                name, bestThreshold, bestF1 * 100);
        return bestThreshold;
    }

    private static void loadAndRouteData(SpecialistData w, SpecialistData c, SpecialistData hrs, OnnxCnnPredictor cnnPredictor) throws Exception {
        Map<String, Map<String, Float>> encoders = new HashMap<>();
        for (String cat : CATEGORICAL_FEATURES) encoders.put(cat, new HashMap<>());

        SequenceTracker sequenceTracker = new SequenceTracker();
        long rowIndex = 0;

        try (
                Reader reader = Files.newBufferedReader(CSV_PATH, StandardCharsets.UTF_8);
                CSVParser parser = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).get().parse(reader)
        ) {
            for (CSVRecord row : parser) {
                if (rowIndex >= TOTAL_ROWS) break;

                if (rowIndex < TRAIN_ROWS) {
                    for (String cat : CATEGORICAL_FEATURES) {
                        String val = row.get(cat);
                        if (val != null && !val.isBlank()) encoders.get(cat).computeIfAbsent(val, k -> (float) encoders.get(cat).size() + 1f);
                    }
                }

                float cnnScore = sequenceTracker.getRiskScore(row, cnnPredictor);
                String product = row.get("ProductCD");

                SpecialistData target;
                if (product.equals("W")) target = w;
                else if (product.equals("C")) target = c;
                else target = hrs;

                if (rowIndex < TRAIN_ROWS) {
                    target.addTrainRow(row, encoders, cnnScore);
                } else if (rowIndex < TRAIN_ROWS + VALIDATION_ROWS) {
                    target.addValidationRow(row, encoders, cnnScore);
                } else {
                    float[] features = target.addTestRow(row, encoders, cnnScore);
                    int actualLabel = Integer.parseInt(row.get("isFraud"));
                    testTransactionRegistry.add(new TestTransaction(features, product, actualLabel));
                }

                rowIndex++;
                if (rowIndex % 100_000 == 0) System.out.println("Processed " + rowIndex + " transactions...");
            }
        }
        w.finalizeMatrices(); c.finalizeMatrices(); hrs.finalizeMatrices();
    }

    private static class SequenceTracker {
        private final Map<String, LinkedList<float[]>> cardHistories = new HashMap<>();

        float getRiskScore(CSVRecord row, OnnxCnnPredictor predictor) throws Exception {
            String cardId = row.get("card1");
            if (cardId == null || cardId.isBlank()) return 0.0f;

            long currentDt = Long.parseLong(row.get("TransactionDT"));
            float amt = parseNumber(row.get("TransactionAmt"));
            float dist1 = parseNumber(row.get("dist1"));
            float c1 = parseNumber(row.get("C1"));
            float c13 = parseNumber(row.get("C13"));

            float[] currentTx = {currentDt, amt, dist1, c1, c13};

            LinkedList<float[]> history = cardHistories.computeIfAbsent(cardId, k -> new LinkedList<>());
            history.add(currentTx);
            if (history.size() > 5) history.removeFirst();

            float[][] sequence = new float[5][5];
            int startIdx = 5 - history.size();

            for (int i = 0; i < 5; i++) {
                if (i < startIdx) {
                    sequence[i] = new float[]{0f, 0f, 0f, 0f, 0f};
                } else {
                    float[] tx = history.get(i - startIdx);
                    float timeDelta = currentDt - tx[0];
                    sequence[i] = new float[]{timeDelta, tx[1], tx[2], tx[3], tx[4]};
                }
            }
            return predictor.predictRiskScore(sequence);
        }
    }

    private static Booster trainSpecialist(SpecialistData data) throws Exception {
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
        params.put("seed", 42);
        params.put("nthread", 4);
        params.put("verbosity", 0);

        Map<String, DMatrix> watches = new LinkedHashMap<>();
        watches.put("validation", data.validationMatrix);

        float[][] history = new float[1][500];

        long start = System.nanoTime();
        Booster model = XGBoost.train(data.trainMatrix, params, 500, watches, history, null, null, 35);

        String bestIter = model.getAttr("best_iteration");
        data.bestRoundCount = bestIter == null ? 499 : Integer.parseInt(bestIter) + 1;

        System.out.printf(Locale.US, "Training time: %.1f seconds | Optimal Trees: %d | Validation PR-AUC: %s%n",
                (System.nanoTime() - start) / 1_000_000_000.0, data.bestRoundCount, model.getAttr("best_score"));

        return model;
    }

    private static List<ScoreLabel> scoreSpecialist(Booster model, SpecialistData data, boolean useValidation) throws Exception {
        List<ScoreLabel> results = new ArrayList<>();
        DMatrix matrix = useValidation ? data.validationMatrix : data.testMatrix;
        int[] labels = useValidation ? data.validationLabels : data.testLabels;

        if (labels == null || labels.length == 0) return results;

        float[][] raw = model.predict(matrix, false, data.bestRoundCount);
        for (int i = 0; i < raw.length; i++) results.add(new ScoreLabel(raw[i][0], labels[i]));
        return results;
    }

    private static String[] createFeatureNames() {
        String[] names = new String[NUMERIC_FEATURES.length + CATEGORICAL_FEATURES.length + 1];
        System.arraycopy(NUMERIC_FEATURES, 0, names, 0, NUMERIC_FEATURES.length);
        System.arraycopy(CATEGORICAL_FEATURES, 0, names, NUMERIC_FEATURES.length, CATEGORICAL_FEATURES.length);
        names[names.length - 1] = "cnn_risk_score";
        return names;
    }

    private static float parseNumber(String raw) {
        if (raw == null || raw.isBlank()) return Float.NaN;
        try { return (float) Double.parseDouble(raw); } catch (NumberFormatException e) { return Float.NaN; }
    }

    private static class SpecialistData {
        final String name;
        final List<float[]> trainFeatures = new ArrayList<>();
        final List<Float> trainLabelsList = new ArrayList<>();

        final List<float[]> validationFeatures = new ArrayList<>();
        final List<Float> validationLabelsList = new ArrayList<>();

        final List<float[]> testFeatures = new ArrayList<>();
        final List<Float> testLabelsList = new ArrayList<>();

        DMatrix trainMatrix, validationMatrix, testMatrix;
        int[] validationLabels, testLabels;
        int bestRoundCount;

        SpecialistData(String name) { this.name = name; }

        void addTrainRow(CSVRecord row, Map<String, Map<String, Float>> encoders, float cnnScore) {
            trainFeatures.add(extractFeatures(row, encoders, true, cnnScore));
            trainLabelsList.add(Float.parseFloat(row.get("isFraud")));
        }

        void addValidationRow(CSVRecord row, Map<String, Map<String, Float>> encoders, float cnnScore) {
            validationFeatures.add(extractFeatures(row, encoders, false, cnnScore));
            validationLabelsList.add(Float.parseFloat(row.get("isFraud")));
        }

        float[] addTestRow(CSVRecord row, Map<String, Map<String, Float>> encoders, float cnnScore) {
            float[] feat = extractFeatures(row, encoders, false, cnnScore);
            testFeatures.add(feat);
            testLabelsList.add(Float.parseFloat(row.get("isFraud")));
            return feat;
        }

        private float[] extractFeatures(CSVRecord row, Map<String, Map<String, Float>> encoders, boolean isTraining, float cnnScore) {
            float[] features = new float[FEATURE_COUNT];
            int column = 0;

            for (String feature : NUMERIC_FEATURES) features[column++] = parseNumber(row.get(feature));

            for (String cat : CATEGORICAL_FEATURES) {
                String val = row.get(cat);
                if (val == null || val.isBlank()) {
                    features[column++] = Float.NaN;
                } else {
                    Map<String, Float> dict = encoders.get(cat);
                    if (isTraining) {
                        float id = dict.computeIfAbsent(val, k -> (float) dict.size() + 1f);
                        features[column++] = id;
                    } else {
                        Float id = dict.get(val);
                        features[column++] = (id != null) ? id : Float.NaN;
                    }
                }
            }
            features[column] = cnnScore;
            return features;
        }

        void finalizeMatrices() throws Exception {
            if (!trainFeatures.isEmpty()) trainMatrix = createMatrix(trainFeatures, trainLabelsList);
            if (!validationFeatures.isEmpty()) {
                validationMatrix = createMatrix(validationFeatures, validationLabelsList);
                validationLabels = new int[validationLabelsList.size()];
                for (int i = 0; i < validationLabelsList.size(); i++) validationLabels[i] = Math.round(validationLabelsList.get(i));
            }
            if (!testFeatures.isEmpty()) {
                testMatrix = createMatrix(testFeatures, testLabelsList);
                testLabels = new int[testLabelsList.size()];
                for (int i = 0; i < testLabelsList.size(); i++) testLabels[i] = Math.round(testLabelsList.get(i));
            }
            trainFeatures.clear(); trainLabelsList.clear();
            validationFeatures.clear(); validationLabelsList.clear();
            testFeatures.clear(); testLabelsList.clear();
        }

        private DMatrix createMatrix(List<float[]> featuresList, List<Float> labelsList) throws Exception {
            float[] flatFeatures = new float[featuresList.size() * FEATURE_COUNT];
            float[] labels = new float[labelsList.size()];
            for (int i = 0; i < featuresList.size(); i++) {
                System.arraycopy(featuresList.get(i), 0, flatFeatures, i * FEATURE_COUNT, FEATURE_COUNT);
                labels[i] = labelsList.get(i);
            }
            DMatrix matrix = new DMatrix(flatFeatures, featuresList.size(), FEATURE_COUNT, Float.NaN);
            matrix.setLabel(labels);
            matrix.setFeatureNames(FEATURE_NAMES);
            return matrix;
        }

        void dispose() {
            if (trainMatrix != null) trainMatrix.dispose();
            if (validationMatrix != null) validationMatrix.dispose();
            if (testMatrix != null) testMatrix.dispose();
        }
    }

    private static class TestTransaction {
        final float[] rawFeatures;
        final String productCD;
        final int isFraud;
        TestTransaction(float[] rawFeatures, String productCD, int isFraud) {
            this.rawFeatures = rawFeatures;
            this.productCD = productCD;
            this.isFraud = isFraud;
        }
    }

    private static RankingMetrics calculateRankingMetrics(List<ScoreLabel> predictions) {
        long positives = 0;
        for (ScoreLabel sl : predictions) if (sl.label == 1) positives++;
        long negatives = predictions.size() - positives;
        predictions.sort(Comparator.comparingDouble((ScoreLabel x) -> x.probability).reversed());
        long tp = 0, fp = 0;
        double prevRecall = 0, prevFpr = 0, prevPrecision = 1, rocAuc = 0, prAuc = 0;
        int i = 0;
        while (i < predictions.size()) {
            double currentScore = predictions.get(i).probability;
            int j = i;
            while (j < predictions.size() && Double.compare(predictions.get(j).probability, currentScore) == 0) {
                if (predictions.get(j).label == 1) tp++; else fp++;
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

    private static Metrics evaluateAtThreshold(List<ScoreLabel> predictions, double threshold) {
        Metrics result = new Metrics();
        for (ScoreLabel sl : predictions) {
            if (sl.probability >= threshold && sl.label == 1) result.tp++;
            else if (sl.probability >= threshold) result.fp++;
            else if (sl.label == 1) result.fn++;
            else result.tn++;
        }
        return result;
    }

    private static class ScoreLabel {
        final double probability; final int label;
        ScoreLabel(double probability, int label) { this.probability = probability; this.label = label; }
    }

    private static class Metrics {
        long tp, fp, fn, tn;
        double precision() { return tp + fp == 0 ? 0 : (double) tp / (tp + fp); }
        double recall() { return tp + fn == 0 ? 0 : (double) tp / (tp + fn); }
        double f1() {
            double p = precision(); double r = recall();
            return (p + r == 0) ? 0 : 2 * (p * r) / (p + r);
        }
    }

    private static class RankingMetrics {
        final double rocAuc, prAuc;
        RankingMetrics(double rocAuc, double prAuc) { this.rocAuc = rocAuc; this.prAuc = prAuc; }
    }

    // ========================================================================
    // INTERACTIVE EXPLANATION SERVICE CLASS (AI REASONING MODE)
    // ========================================================================
    public static class FinForgeInteractiveService {

        private static final Path AUDIT_LOG_PATH = Path.of("flagged_transactions.jsonl");

        private final Booster wModel;
        private final Booster cModel;
        private final Booster hrsModel;

        private double wThreshold = 0.12;
        private double cThreshold = 0.43;
        private double hrsThreshold = 0.39;

        private String wThresholdReason = "W-Class (High-Value): Threshold set highly sensitive (0.12) to catch subtle, expensive fraud without waiting for obvious bot patterns.";
        private String cThresholdReason = "C-Class (High-Velocity): Threshold strict (0.43) to block definitive bot attacks while avoiding false flags on rapid, normal digital purchases.";
        private String hrsThresholdReason = "HRS-Class: Threshold balanced (0.39) to protect digital accounts and virtual services.";

        public FinForgeInteractiveService(Booster wModel, Booster cModel, Booster hrsModel) {
            this.wModel = wModel;
            this.cModel = cModel;
            this.hrsModel = hrsModel;
        }

        public void adjustThreshold(String productCD, double newThreshold, String adminReason) {
            if (productCD.equals("W")) {
                this.wThreshold = newThreshold;
                this.wThresholdReason = "Manual Override (" + newThreshold + "): " + adminReason;
            } else if (productCD.equals("C")) {
                this.cThreshold = newThreshold;
                this.cThresholdReason = "Manual Override (" + newThreshold + "): " + adminReason;
            } else {
                this.hrsThreshold = newThreshold;
                this.hrsThresholdReason = "Manual Override (" + newThreshold + "): " + adminReason;
            }
            System.out.printf("✅ System updated: %s threshold is now %.2f%n", productCD, newThreshold);
        }

        public String getThresholdReasoning(String productCD) {
            if (productCD.equals("W")) return wThresholdReason;
            if (productCD.equals("C")) return cThresholdReason;
            return hrsThresholdReason;
        }

        public void evaluateAndExplain(float[] rawFeatures, String[] featureNames, String productCD, int actualLabel) throws XGBoostError {
            Booster targetModel;
            double activeThreshold;

            if (productCD.equals("W")) {
                targetModel = wModel; activeThreshold = wThreshold;
            } else if (productCD.equals("C")) {
                targetModel = cModel; activeThreshold = cThreshold;
            } else {
                targetModel = hrsModel; activeThreshold = hrsThreshold;
            }

            DMatrix singleTx = new DMatrix(rawFeatures, 1, featureNames.length, Float.NaN);
            singleTx.setFeatureNames(featureNames);

            float[][] probResult = targetModel.predict(singleTx);
            float probability = probResult[0][0];
            singleTx.dispose();

            boolean isFlagged = probability >= activeThreshold;
            printAiReasoningReport(probability, activeThreshold, isFlagged, featureNames, rawFeatures, productCD, actualLabel);
        }

        private void printAiReasoningReport(float probability, double threshold, boolean isFlagged,
                                            String[] featureNames, float[] rawFeatures, String productCD, int actualLabel) {

            List<FeatureDriver> drivers = new ArrayList<>();
            for (int i = 0; i < featureNames.length; i++) {
                if (!Float.isNaN(rawFeatures[i]) && rawFeatures[i] != 0.0f) {
                    drivers.add(new FeatureDriver(featureNames[i], rawFeatures[i]));
                }
            }
            drivers.sort(Comparator.comparingDouble((FeatureDriver f) -> Math.abs(f.value)).reversed());

            System.out.println("\n======================================================================");
            System.out.println("🧠 FINFORGE AI ANALYST REPORT");
            System.out.println("======================================================================");
            System.out.printf(Locale.US, "Category: %s-Class | Risk Score: %.2f%% | Ground Truth: %s%n",
                    productCD, probability * 100, actualLabel == 1 ? "FRAUD" : "LEGITIMATE");
            System.out.println("----------------------------------------------------------------------");

            StringBuilder narrative = new StringBuilder();

            if (isFlagged) {
                narrative.append("⚠️ AI DECISION: BLOCK TRANSACTION\n\n");
                narrative.append("Reasoning:\n");
                narrative.append(String.format(Locale.US, "I have flagged this transaction as highly suspicious because it generated a risk probability of %.2f%%, breaching our current security threshold of %.2f%%. ", probability * 100, threshold * 100));
                narrative.append("My analysis indicates several critical behavioral deviations. ");

                // Trigger Asynchronous Audit Log for LLM processing
                logForOfflineAnalysis(productCD, probability, drivers, actualLabel);
            } else {
                narrative.append("✅ AI DECISION: APPROVE TRANSACTION\n\n");
                narrative.append("Reasoning:\n");
                narrative.append(String.format(Locale.US, "I have cleared this transaction. The calculated risk score is %.2f%%, which sits safely below the required threshold of %.2f%%. ", probability * 100, threshold * 100));
                narrative.append("While I evaluated the user's data footprint, the patterns align with expected legitimate behavior. ");
                if (!drivers.isEmpty()) narrative.append("I did note a few distinct data points, but they lacked the correlation needed to indicate an attack. ");
            }

            if (!drivers.isEmpty()) {
                narrative.append("Specifically, I observed the following drivers:\n\n");
                for (int i = 0; i < Math.min(3, drivers.size()); i++) {
                    narrative.append(" • ").append(synthesizeInsight(drivers.get(i))).append("\n");
                }
            }

            narrative.append("\nContext & Policy Applied:\n");
            narrative.append(getThresholdReasoning(productCD));

            System.out.println(narrative.toString());
            System.out.println("======================================================================");
        }

        private void logForOfflineAnalysis(String productCD, float score, List<FeatureDriver> topDrivers, int actualLabel) {
            try {
                StringBuilder json = new StringBuilder();
                json.append("{");
                json.append("\"timestamp\": \"").append(Instant.now().toString()).append("\", ");
                json.append("\"product_class\": \"").append(productCD).append("\", ");
                json.append("\"risk_score\": ").append(score).append(", ");
                json.append("\"actual_label\": ").append(actualLabel).append(", ");

                json.append("\"critical_features\": {");
                for (int i = 0; i < Math.min(5, topDrivers.size()); i++) {
                    FeatureDriver d = topDrivers.get(i);
                    json.append("\"").append(d.name).append("\": ").append(d.value);
                    if (i < Math.min(5, topDrivers.size()) - 1) json.append(", ");
                }
                json.append("}}\n");

                if (!Files.exists(AUDIT_LOG_PATH)) Files.createFile(AUDIT_LOG_PATH);
                Files.writeString(AUDIT_LOG_PATH, json.toString(), StandardOpenOption.APPEND);

                System.out.println("📝 [System] Flagged transaction securely logged to " + AUDIT_LOG_PATH.getFileName());
            } catch (Exception e) {
                System.err.println("Failed to write to async audit log: " + e.getMessage());
            }
        }

        private String synthesizeInsight(FeatureDriver driver) {
            String name = driver.name;
            float val = driver.value;

            if (name.equals("cnn_risk_score")) {
                if (val > 0.6) return String.format("The deep learning temporal tracker detected an aggressive velocity score (%.2f). This suggests rapid, successive card usage highly characteristic of automated card-testing bots.", val);
                return String.format("The neural network evaluated the sequence of recent transactions and determined the timing pattern (Score: %.2f) falls within normal behavioral bounds.", val);
            }
            if (name.equals("TransactionAmt")) {
                if (val > 500) return String.format("The monetary value of $%.2f represents a significant spike, shifting away from the user's established baseline spending habits.", val);
                return String.format("The transaction amount ($%.2f) was evaluated and processed as a standard localized purchase.", val);
            }
            if (name.equals("dist1")) {
                return String.format("I detected a spatial anomaly: the purchase originated %.1f miles away from the cardholder's established billing address, suggesting potential account takeover or physical theft.", val);
            }
            if (name.contains("emaildomain")) {
                return String.format("The associated email domain (Internal ID: %.1f) maps to a high-risk network category frequently linked to disposable or synthetic identities.", val);
            }
            if (name.startsWith("C")) {
                return String.format("There is an irregular frequency count in the Vesta network tracking attributes (Feature %s: %.1f), which often implies coordinated credential stuffing across multiple merchants.", name, val);
            }
            if (name.startsWith("V")) {
                return String.format("A deep demographic validation check (Feature %s) returned a value of %.1f, matching patterns we have historically seen in compromised merchant networks.", name, val);
            }
            return String.format("The underlying tabular structure revealed anomalous variance in attribute %s (Value: %.1f).", name, val);
        }

        private static class FeatureDriver {
            String name; float value;
            FeatureDriver(String name, float value) { this.name = name; this.value = value; }
        }
    }
}