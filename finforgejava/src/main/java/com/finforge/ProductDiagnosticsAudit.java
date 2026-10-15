package com.finforge;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;

public class ProductDiagnosticsAudit {

    private static final Path XGB_PREDICTIONS = Path.of(
            "reports", "xgboost_test_predictions.csv"
    );

    public static void main(String[] args) throws IOException {
        System.out.println("=== FINFORGE PRODUCT DIAGNOSTICS AUDIT ===");

        if (!Files.exists(XGB_PREDICTIONS)) {
            System.err.println("Predictions file not found: " + XGB_PREDICTIONS.toAbsolutePath());
            System.err.println("Please run FraudXGBoost.java first.");
            return;
        }

        // TreeMap automatically sorts the products alphabetically
        Map<String, ProductStats> statsMap = new TreeMap<>();

        try (
                Reader reader = Files.newBufferedReader(XGB_PREDICTIONS, StandardCharsets.UTF_8);
                CSVParser parser = CSVFormat.DEFAULT.builder()
                        .setHeader()
                        .setSkipHeaderRecord(true)
                        .get()
                        .parse(reader)
        ) {
            for (CSVRecord row : parser) {
                String product = row.get("ProductCD");
                int actual = Integer.parseInt(row.get("actualFraud"));
                int predicted = Integer.parseInt(row.get("predictedFraud"));

                statsMap.computeIfAbsent(product, k -> new ProductStats(product))
                        .record(actual, predicted);
            }
        }

        // Print the detailed tabular report
        System.out.printf(Locale.US, "%-9s | %-12s | %-12s | %-10s | %-6s | %-8s | %-9s | %-9s%n",
                "Product", "Total Trans.", "Actual Fraud", "Fraud Rate", "TP", "Alerts", "Recall", "Precision");
        System.out.println("-".repeat(95));

        long totalTrans = 0, totalFraud = 0, totalTp = 0, totalAlerts = 0;

        for (ProductStats stats : statsMap.values()) {
            totalTrans += stats.total;
            totalFraud += stats.actualFraud;
            totalTp += stats.tp;
            totalAlerts += stats.alerts();

            System.out.printf(Locale.US, "%-9s | %-12d | %-12d | %-9.2f%% | %-6d | %-8d | %-8.2f%% | %-8.2f%%%n",
                    stats.product,
                    stats.total,
                    stats.actualFraud,
                    stats.fraudRate() * 100,
                    stats.tp,
                    stats.alerts(),
                    stats.recall() * 100,
                    stats.precision() * 100
            );
        }

        System.out.println("-".repeat(95));
        double overallRecall = totalFraud == 0 ? 0 : (double) totalTp / totalFraud * 100;
        double overallPrecision = totalAlerts == 0 ? 0 : (double) totalTp / totalAlerts * 100;

        System.out.printf(Locale.US, "%-9s | %-12d | %-12d | %-9.2f%% | %-6d | %-8d | %-8.2f%% | %-8.2f%%%n",
                "OVERALL", totalTrans, totalFraud, (double)totalFraud/totalTrans*100, totalTp, totalAlerts, overallRecall, overallPrecision);
        System.out.println("\nNote: 'Alerts' = True Positives (TP) + False Positives (FP).");
    }

    private static class ProductStats {
        final String product;
        long total = 0;
        long actualFraud = 0;
        long tp = 0;
        long fp = 0;
        long fn = 0;
        long tn = 0;

        ProductStats(String product) {
            this.product = product;
        }

        void record(int actual, int predicted) {
            total++;
            if (actual == 1) actualFraud++;

            if (predicted == 1 && actual == 1) tp++;
            else if (predicted == 1 && actual == 0) fp++;
            else if (predicted == 0 && actual == 1) fn++;
            else tn++;
        }

        long alerts() { return tp + fp; }
        double fraudRate() { return total == 0 ? 0 : (double) actualFraud / total; }
        double recall() { return actualFraud == 0 ? 0 : (double) tp / actualFraud; }
        double precision() { return alerts() == 0 ? 0 : (double) tp / alerts(); }
    }
}