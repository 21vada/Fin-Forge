package com.finforge;

import ai.onnxruntime.*;
import java.nio.FloatBuffer;
import java.util.Collections;

public class OnnxAttentionPredictor implements AutoCloseable {

    private static final String MODEL_PATH = "models/fraud_transformer.onnx";
    private final OrtEnvironment env;
    private final OrtSession session;

    public OnnxAttentionPredictor() throws Exception {
        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(MODEL_PATH, new OrtSession.SessionOptions());
    }

    /**
     * Evaluates a [5, 5] temporal rolling sequence (5 time steps x 5 features)
     * and returns the Transformer attention risk score [0.0 - 1.0].
     */
    public float predictRiskScore(float[][] sequenceWindow) throws Exception {
        // Flatten 2D array [5][5] into 1D array of 25 float values
        float[] flattened = new float[25];
        int idx = 0;
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                flattened[idx++] = sequenceWindow[i][j];
            }
        }

        // Tensor shape: [batch_size=1, sequence_length=5, feature_dim=5]
        long[] shape = new long[]{1, 5, 5};
        FloatBuffer buffer = FloatBuffer.wrap(flattened);

        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, buffer, shape)) {
            // Run inference against ONNX input node named 'sequence_input'
            try (OrtSession.Result result = session.run(Collections.singletonMap("sequence_input", inputTensor))) {
                float[][] outputArray = (float[][]) result.get(0).getValue();
                return outputArray[0][0]; // Output risk probability from Sigmoid head
            }
        }
    }

    @Override
    public void close() throws Exception {
        if (session != null) session.close();
        if (env != null) env.close();
    }
}