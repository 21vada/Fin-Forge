
package com.finforge;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

public class DatasetAudit {

    private static final Path DATA_DIR = Path.of(
            "C:/Users/pc/Documents/fin forge/ieee-fraud-detection"
    );

    private static final Path TRANSACTION_FILE =
            DATA_DIR.resolve("train_transaction.csv");

    private static final Path IDENTITY_FILE =
            DATA_DIR.resolve("train_identity.csv");

    private static final String[] AUDITED_FEATURES = {
            "TransactionAmt",
            "ProductCD",
            "card1",
            "addr1",
            "dist1",
            "C1",
            "V1",
            "V2"
    };

    public static void main(String[] args) throws IOException {

        requireFile(TRANSACTION_FILE);
        requireFile(IDENTITY_FILE);

        System.out.println("Loading identity IDs...");
        IdentitySummary identity = loadIdentityIds();

        System.out.println("Scanning all transactions...");

        Map<String, Counts> productStats = new LinkedHashMap<>();
        Map<String, Long> missingCounts = new LinkedHashMap<>();

        for (String feature : AUDITED_FEATURES) {
            missingCounts.put(feature, 0L);
        }

        long totalRows = 0;
        long fraudRows = 0;
        long identityMatches = 0;
        long timeMissing = 0;
        long outOfOrderRows = 0;

        long minTime = Long.MAX_VALUE;
        long maxTime = Long.MIN_VALUE;
        long previousTime = Long.MIN_VALUE;

        try (
                Reader reader = Files.newBufferedReader(
                        TRANSACTION_FILE, StandardCharsets.UTF_8
                );
                CSVParser parser = CSVFormat.DEFAULT.builder()
                        .setHeader()
                        .setSkipHeaderRecord(true)
                        .get()
                        .parse(reader)
        ) {
            for (CSVRecord row : parser) {

                totalRows++;

                String transactionId = row.get("TransactionID");
                int label = Integer.parseInt(row.get("isFraud"));

                if (label == 1) {
                    fraudRows++;
                }

                if (identity.transactionIds.contains(transactionId)) {
                    identityMatches++;
                }

                // Track the distribution by product category.
                String product = row.get("ProductCD");

                if (product == null || product.isBlank()) {
                    product = "(missing)";
                }

                productStats
                        .computeIfAbsent(product, key -> new Counts())
                        .add(label);

                // Measure missingness in selected transaction features.
                for (String feature : AUDITED_FEATURES) {
                    String value = row.get(feature);

                    if (value == null || value.isBlank()) {
                        missingCounts.merge(feature, 1L, Long::sum);
                    }
                }

                // TransactionDT is an offset, not a calendar timestamp.
                String rawTime = row.get("TransactionDT");

                if (rawTime == null || rawTime.isBlank()) {
                    timeMissing++;
                    continue;
                }

                long time = (long) Double.parseDouble(rawTime);

                minTime = Math.min(minTime, time);
                maxTime = Math.max(maxTime, time);

                if (previousTime != Long.MIN_VALUE
                        && time < previousTime) {
                    outOfOrderRows++;
                }

                previousTime = time;
            }
        }

        StringBuilder report = new StringBuilder();

        report.append("=== FINFORGE DATA AUDIT ===\n\n");

        report.append("TRANSACTION DATA\n");
        report.append("------------------------------\n");
        report.append("Transaction rows: ")
                .append(totalRows).append('\n');
        report.append("Normal transactions: ")
                .append(totalRows - fraudRows).append('\n');
        report.append("Fraudulent transactions: ")
                .append(fraudRows).append('\n');
        report.append("Fraud prevalence: ")
                .append(percent(fraudRows, totalRows)).append('\n');

        report.append("Minimum TransactionDT: ")
                .append(minTime == Long.MAX_VALUE ? "N/A" : minTime)
                .append('\n');
        report.append("Maximum TransactionDT: ")
                .append(maxTime == Long.MIN_VALUE ? "N/A" : maxTime)
                .append('\n');
        report.append("Rows with missing TransactionDT: ")
                .append(timeMissing).append('\n');
        report.append("Rows earlier than preceding row: ")
                .append(outOfOrderRows).append('\n');

        report.append("\nIDENTITY COVERAGE\n");
        report.append("------------------------------\n");
        report.append("Identity file rows: ")
                .append(identity.rows).append('\n');
        report.append("Unique identity transaction IDs: ")
                .append(identity.transactionIds.size()).append('\n');
        report.append("Transactions with identity match: ")
                .append(identityMatches).append('\n');
        report.append("Identity match rate: ")
                .append(percent(identityMatches, totalRows)).append('\n');

        report.append("\nFRAUD RATE BY PRODUCT\n");
        report.append("------------------------------\n");

        for (Map.Entry<String, Counts> entry
                : productStats.entrySet()) {

            Counts counts = entry.getValue();

            report.append(String.format(
                    Locale.US,
                    "%-12s | Rows: %7d | Fraud: %6d | Rate: %6.2f%%%n",
                    entry.getKey(),
                    counts.total,
                    counts.fraud,
                    counts.fraudRate() * 100
            ));
        }

        report.append("\nFEATURE MISSINGNESS\n");
        report.append("------------------------------\n");

        for (Map.Entry<String, Long> entry
                : missingCounts.entrySet()) {

            report.append(String.format(
                    Locale.US,
                    "%-18s | Missing: %7d | Rate: %6.2f%%%n",
                    entry.getKey(),
                    entry.getValue(),
                    percentage(entry.getValue(), totalRows)
            ));
        }

        Path reportPath = Path.of("reports", "dataset_audit.txt");
        Files.createDirectories(reportPath.getParent());

        Files.writeString(
                reportPath,
                report.toString(),
                StandardCharsets.UTF_8
        );

        System.out.println();
        System.out.println(report);
        System.out.println(
                "Report saved to: " + reportPath.toAbsolutePath()
        );
    }

    private static IdentitySummary loadIdentityIds()
            throws IOException {

        Set<String> ids = new HashSet<>();
        long rows = 0;

        try (
                Reader reader = Files.newBufferedReader(
                        IDENTITY_FILE, StandardCharsets.UTF_8
                );
                CSVParser parser = CSVFormat.DEFAULT.builder()
                        .setHeader()
                        .setSkipHeaderRecord(true)
                        .get()
                        .parse(reader)
        ) {
            for (CSVRecord row : parser) {
                ids.add(row.get("TransactionID"));
                rows++;
            }
        }

        return new IdentitySummary(rows, ids);
    }

    private static void requireFile(Path path) throws IOException {
        if (!Files.isRegularFile(path)) {
            throw new IOException(
                    "Required dataset file not found: "
                            + path.toAbsolutePath()
            );
        }
    }

    private static String percent(long numerator, long denominator) {
        return String.format(
                Locale.US,
                "%.2f%%",
                percentage(numerator, denominator)
        );
    }

    private static double percentage(long numerator, long denominator) {
        return denominator == 0
                ? 0
                : 100.0 * numerator / denominator;
    }

    private static class IdentitySummary {
        final long rows;
        final Set<String> transactionIds;

        IdentitySummary(long rows, Set<String> transactionIds) {
            this.rows = rows;
            this.transactionIds = transactionIds;
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

        double fraudRate() {
            return total == 0 ? 0 : (double) fraud / total;
        }
    }
}
