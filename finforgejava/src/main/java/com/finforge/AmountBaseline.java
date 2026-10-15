
package com.finforge;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

public class AmountBaseline {

    public static void main(String[] args) throws IOException {

        Path csvPath = Path.of(
                "C:/Users/pc/Documents/fin forge/ieee-fraud-detection/train_transaction.csv"
        );

        Map<String, AmountStats> statsByProduct =
                new LinkedHashMap<>();

        long trainingRows = 0;
        long evaluatedRows = 0;

        long truePositives = 0;
        long falsePositives = 0;
        long falseNegatives = 0;
        long trueNegatives = 0;

        try (
                Reader reader = Files.newBufferedReader(
                        csvPath, StandardCharsets.UTF_8
                );
                CSVParser parser = CSVFormat.DEFAULT.builder()
                        .setHeader()
                        .setSkipHeaderRecord(true)
                        .get()
                        .parse(reader)
        ) {
            for (CSVRecord row : parser) {

                String product = row.get("ProductCD");
                double amount = Double.parseDouble(
                        row.get("TransactionAmt")
                );

                // Phase 1: Learn from the first 8,000 rows.
                if (trainingRows < 8000) {

                    statsByProduct
                            .computeIfAbsent(
                                    product, key -> new AmountStats()
                            )
                            .add(amount);

                    trainingRows++;
                    continue;
                }

                // Phase 2: Evaluate the next 2,000 rows.
                if (evaluatedRows >= 2000) {
                    break;
                }

                int actualFraud = Integer.parseInt(
                        row.get("isFraud")
                );

                AmountStats stats = statsByProduct.get(product);
                boolean flagged = false;

                if (stats != null) {
                    double stdDev = stats.getStandardDeviation();

                    if (stdDev > 0) {
                        double distance = Math.abs(
                                amount - stats.mean
                        );

                        flagged = distance >= 3 * stdDev;
                    }
                }

                // Compare our prediction with the true label.
                if (flagged && actualFraud == 1) {
                    truePositives++;
                } else if (flagged) {
                    falsePositives++;
                } else if (actualFraud == 1) {
                    falseNegatives++;
                } else {
                    trueNegatives++;
                }

                evaluatedRows++;
            }
        }

        long alerts = truePositives + falsePositives;

        System.out.println("\n=== FINFORGE FRAUD REPORT ===");
        System.out.println("Training rows:   " + trainingRows);
        System.out.println("Evaluation rows: " + evaluatedRows);
        System.out.println("Alerts:          " + alerts);
        System.out.println("True positives:  " + truePositives);
        System.out.println("False positives: " + falsePositives);
        System.out.println("False negatives: " + falseNegatives);
        System.out.println("True negatives:  " + trueNegatives);

        System.out.println("Precision: " +
                percentage(truePositives, alerts));

        System.out.println("Recall:    " +
                percentage(
                        truePositives,
                        truePositives + falseNegatives
                ));
    }

    private static String percentage(long numerator, long denominator) {
        if (denominator == 0) {
            return "N/A";
        }

        return String.format(
                "%.2f%%",
                100.0 * numerator / denominator
        );
    }

    private static class AmountStats {

        long count = 0;
        double mean = 0;
        double squaredDifferences = 0;

        void add(double value) {
            count++;

            double difference = value - mean;
            mean += difference / count;

            double updatedDifference = value - mean;
            squaredDifferences += difference * updatedDifference;
        }

        double getStandardDeviation() {
            if (count == 0) {
                return 0;
            }

            return Math.sqrt(squaredDifferences / count);
        }
    }
}
