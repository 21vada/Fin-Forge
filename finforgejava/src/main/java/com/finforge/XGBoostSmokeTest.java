
package com.finforge;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;

import java.util.HashMap;
import java.util.Map;

public class XGBoostSmokeTest {

    public static void main(String[] args) throws Exception {

        // Eight synthetic examples, with two numerical features each.
        // Feature 1 could represent amount; feature 2 could represent
        // another transaction signal. These are not real fraud data.
        float[] features = {
                10, 1,
                20, 1,
                15, 2,
                25, 2,
                800, 8,
                900, 9,
                750, 8,
                950, 9
        };

        float[] labels = {
                0, 0, 0, 0,
                1, 1, 1, 1
        };

        DMatrix trainingData =
                new DMatrix(features, 8, 2, Float.NaN);

        trainingData.setLabel(labels);

        Map<String, Object> parameters = new HashMap<>();
        parameters.put("objective", "binary:logistic");
        parameters.put("eval_metric", "logloss");
        parameters.put("max_depth", 2);
        parameters.put("eta", 0.3);
        parameters.put("tree_method", "hist");
        parameters.put("nthread", 2);
        parameters.put("seed", 42);

        System.out.println("Training a tiny XGBoost model...");

        Booster model = XGBoost.train(
                trainingData,
                parameters,
                10,
                new HashMap<>(),
                null,
                null
        );

        float[][] predictions = model.predict(trainingData);

        System.out.println("\n=== XGBOOST SMOKE TEST ===");

        for (int i = 0; i < predictions.length; i++) {
            System.out.printf(
                    "Example %d | Actual: %d | Predicted fraud probability: %.4f%n",
                    i + 1,
                    (int) labels[i],
                    predictions[i][0]
            );
        }

        System.out.println("\nXGBoost training and prediction succeeded.");
    }
}
