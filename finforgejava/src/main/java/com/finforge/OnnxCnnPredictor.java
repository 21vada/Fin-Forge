package com.finforge;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Locale;

public class OnnxCnnPredictor implements AutoCloseable {

    private static final Path ONNX_MODEL_PATH = Path.of("models", "fraud_cnn.onnx");
    private static final int TIMESTEPS = 5;
    private static final int FEATURES = 5;

    // ========================================================================
    // TODO: REPLACE THESE TWO LINES WITH THE OUTPUT FROM THE PYTHON SCRIPT


    private static final float[] MEANS = { 405744.7061597073f, 125.70438460470768f, 45.65163517030666f, 13.399437969144882f, 31.205164837237373f, 346136.6411984648f, 127.73401491619501f, 46.031972577420696f, 13.572250669533402f, 31.51315963185093f, 271321.9285249519f, 129.84346016298835f, 46.664556693448006f, 13.612080834370639f, 31.729892780355325f, 167416.84249462487f, 132.44051587808713f, 47.20929048319565f, 13.835176907698692f, 32.202630511863f, 0.0f, 134.96037703479422f, 47.92534231073894f, 13.987222285089208f, 32.598711383953834f };
    private static final float[] SCALES = { 1161593.9683712078f, 231.3953018159147f, 238.16371522892575f, 130.6070514700007f, 126.95424348879384f, 1054114.7294274985f, 233.2898702815022f, 238.76795988007441f, 132.0132573363843f, 127.66303572541175f, 916717.4879061949f, 233.8144906362574f, 239.91703224768014f, 131.63764451995078f, 127.78105264923165f, 697357.8862566623f, 236.4645190698059f, 240.78590243673128f, 132.18034349313731f, 128.54582075048063f, 1.0f, 237.34850830952033f, 243.5039355697088f, 131.59283239686485f, 128.65153808275434f };

    // ========================================================================

    private final OrtEnvironment env;
    private final OrtSession session;

    public OnnxCnnPredictor() throws OrtException {
        if (!Files.exists(ONNX_MODEL_PATH)) {
            throw new IllegalStateException("ONNX model file not found at: " + ONNX_MODEL_PATH.toAbsolutePath());
        }
        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(ONNX_MODEL_PATH.toString(), new OrtSession.SessionOptions());
        System.out.println("[ONNX] Loaded fraud_cnn.onnx successfully into Java Runtime.");
    }

    /**
     * Scores a 5-step transaction window.
     *
     * @param rawSequence 2D array of shape [5 timesteps][5 features]
     *                    Features per step: [TimeDelta, TransactionAmt, dist1, C1, C13]
     * @return CNN Fraud Risk Probability between 0.0 and 1.0
     */
    public float predictRiskScore(float[][] rawSequence) throws OrtException {
        if (rawSequence.length != TIMESTEPS || rawSequence[0].length != FEATURES) {
            throw new IllegalArgumentException(String.format(
                    "Input sequence must be [%d][%d], but got [%d][%d]",
                    TIMESTEPS, FEATURES, rawSequence.length, rawSequence[0].length
            ));
        }

        // PyTorch Conv1D tensor shape: [Batch_Size, Channels (Features), Length (Timesteps)]
        float[][][] inputTensorData = new float[1][FEATURES][TIMESTEPS];

        for (int t = 0; t < TIMESTEPS; t++) {
            for (int f = 0; f < FEATURES; f++) {
                // Apply Standard Scaler: (RawValue - Mean) / StandardDeviation
                float scaledValue = (rawSequence[t][f] - MEANS[f]) / SCALES[f];

                // Transpose to match PyTorch's expected [Features][Timesteps] orientation
                inputTensorData[0][f][t] = scaledValue;
            }
        }

        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputTensorData)) {
            try (OrtSession.Result results = session.run(Collections.singletonMap("input", inputTensor))) {
                float[][] rawLogits = (float[][]) results.get(0).getValue();

                // Convert raw neural network logit output to Sigmoid probability
                return sigmoid(rawLogits[0][0]);
            }
        }
    }

    private float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }

    @Override
    public void close() {
        try {
            if (session != null) session.close();
            if (env != null) env.close();
        } catch (OrtException e) {
            System.err.println("Error closing ONNX runtime resources: " + e.getMessage());
        }
    }

    // Verification runner
    public static void main(String[] args) throws Exception {
        try (OnnxCnnPredictor predictor = new OnnxCnnPredictor()) {
            float[][] sampleSequence = new float[][] {
                    { 10000f,  15.0f,  0.0f, 1.0f, 0.0f },
                    {  5000f,  20.0f,  0.0f, 2.0f, 0.0f },
                    {  1200f,  50.0f,  0.0f, 3.0f, 1.0f },
                    {   300f, 200.0f,  0.0f, 5.0f, 1.0f },
                    {    10f, 2500.0f, 0.0f, 8.0f, 2.0f }
            };

            float score = predictor.predictRiskScore(sampleSequence);
            System.out.printf(Locale.US, "Sample Sequence Risk Score: %.4f (%.2f%%)%n", score, score * 100);
        }
    }
}