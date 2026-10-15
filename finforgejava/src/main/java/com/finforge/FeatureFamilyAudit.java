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

public class FeatureFamilyAudit {

    private static final Path CSV_PATH = Path.of(
            "C:/Users/pc/Documents/fin forge/ieee-fraud-detection/train_transaction.csv"
    );

    // Key categorical features from different dataset families
    private static final String[] TARGET_FEATURES = {
            "P_emaildomain", // Purchaser email
            "R_emaildomain", // Recipient email
            "card4",         // Card network (Visa, Mastercard, etc.)
            "card6",         // Card type (Credit, Debit)
            "M4"             // Vesta match indicator
    };

    public static void main(String[] args) throws IOException {

        System.out.println("=== FINFORGE FEATURE FAMILY AUDIT ===\n");

        if (!Files.exists(CSV_PATH)) {
            System.err.println("Dataset not found: " + CSV_PATH.toAbsolutePath());
            return;
        }

        FeatureStats[] stats = new FeatureStats[TARGET_FEATURES.length];
        for (int i = 0; i < TARGET_FEATURES.length; i++) {
            stats[i] = new FeatureStats(TARGET_FEATURES[i]);
        }

        long totalRows = 0;

        try (
                Reader reader = Files.newBufferedReader(CSV_PATH, StandardCharsets.UTF_8);
                CSVParser parser = CSVFormat.DEFAULT.builder()
                        .setHeader()
                        .setSkipHeaderRecord(true)
                        .get()
                        .parse(reader)
        ) {
            System.out.println("Scanning transaction dataset...");

            for (CSVRecord row : parser) {
                totalRows++;
                int isFraud = Integer.parseInt(row.get("isFraud"));

                for (FeatureStats stat : stats) {
                    String value = row.get(stat.name);
                    boolean isMissing = (value == null || value.isBlank());
                    stat.record(isMissing, isFraud);
                }

                if (totalRows % 100_000 == 0) {
                    System.out.println("Processed " + totalRows + " rows...");
                }
            }
        }

        System.out.println("\n--- AUDIT RESULTS ---");
        System.out.println("Total Transactions: " + totalRows + "\n");

        for (FeatureStats stat : stats) {
            stat.printReport();
        }
    }

    private static class FeatureStats {
        final String name;
        long missingCount = 0;
        long missingFraudCount = 0;
        long presentCount = 0;
        long presentFraudCount = 0;

        FeatureStats(String name) {
            this.name = name;
        }

        void record(boolean isMissing, int isFraud) {
            if (isMissing) {
                missingCount++;
                if (isFraud == 1) missingFraudCount++;
            } else {
                presentCount++;
                if (isFraud == 1) presentFraudCount++;
            }
        }

        void printReport() {
            long total = missingCount + presentCount;
            double missingRate = (double) missingCount / total * 100;

            double missingFraudRate = missingCount == 0 ? 0 : (double) missingFraudCount / missingCount * 100;
            double presentFraudRate = presentCount == 0 ? 0 : (double) presentFraudCount / presentCount * 100;

            System.out.printf(Locale.US, "[ Feature: %s ]%n", name);
            System.out.printf(Locale.US, "  Missing: %d (%.2f%% of total)%n", missingCount, missingRate);
            System.out.printf(Locale.US, "  Fraud rate when MISSING: %.2f%%%n", missingFraudRate);
            System.out.printf(Locale.US, "  Fraud rate when PRESENT: %.2f%%%n", presentFraudRate);
            System.out.println("  --------------------------------------");
        }
    }
}