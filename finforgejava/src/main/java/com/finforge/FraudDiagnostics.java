
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

public class FraudDiagnostics {

    private static final Path CSV_PATH = Path.of(
            "C:/Users/pc/Documents/fin forge/ieee-fraud-detection/train_transaction.csv"
    );

    public static void main(String[] args) throws IOException {

        Map<String, Counts> byAmount = new LinkedHashMap<>();
        Map<String, Counts> byProduct = new LinkedHashMap<>();

        String[] bands = {
                "0-25",
                "25-50",
                "50-100",
                "100-250",
                "250-500",
                "500+"
        };

        for (String band : bands) {
            byAmount.put(band, new Counts());
        }

        try (
                Reader reader = Files.newBufferedReader(
                        CSV_PATH, StandardCharsets.UTF_8
                );
                CSVParser parser = CSVFormat.DEFAULT.builder()
                        .setHeader()
                        .setSkipHeaderRecord(true)
                        .get()
                        .parse(reader)
        ) {
            int rowsRead = 0;

            for (CSVRecord row : parser) {

                if (rowsRead >= 8000) {
                    break;
                }

                double amount = Double.parseDouble(
                        row.get("TransactionAmt")
                );

                String product = row.get("ProductCD");
                int fraud = Integer.parseInt(row.get("isFraud"));

                String band;

                if (amount < 25) {
                    band = "0-25";
                } else if (amount < 50) {
                    band = "25-50";
                } else if (amount < 100) {
                    band = "50-100";
                } else if (amount < 250) {
                    band = "100-250";
                } else if (amount < 500) {
                    band = "250-500";
                } else {
                    band = "500+";
                }

                byAmount.get(band).add(fraud);

                byProduct
                        .computeIfAbsent(product, key -> new Counts())
                        .add(fraud);

                rowsRead++;
            }

            System.out.println("=== FINFORGE FRAUD DIAGNOSTICS ===");
            System.out.println("Training rows analysed: " + rowsRead);

            printReport("FRAUD RATE BY TRANSACTION AMOUNT", byAmount);
            printReport("FRAUD RATE BY PRODUCT CATEGORY", byProduct);
        }
    }

    private static void printReport(
            String title,
            Map<String, Counts> groups
    ) {
        System.out.println("\n" + title);
        System.out.println("----------------------------------------------");

        for (Map.Entry<String, Counts> entry : groups.entrySet()) {
            Counts c = entry.getValue();

            double rate = c.total == 0
                    ? 0
                    : 100.0 * c.fraud / c.total;

            System.out.printf(
                    "%-12s | Transactions: %5d | Fraud: %4d | Rate: %.2f%%%n",
                    entry.getKey(),
                    c.total,
                    c.fraud,
                    rate
            );
        }
    }

    private static class Counts {
        long total = 0;
        long fraud = 0;

        void add(int label) {
            total++;

            if (label == 1) {
                fraud++;
            }
        }
    }
}
