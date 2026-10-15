
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

public class TemporalSplitAudit {

    private static final Path CSV_PATH = Path.of(
            "C:/Users/pc/Documents/fin forge/ieee-fraud-detection/train_transaction.csv"
    );

    public static void main(String[] args) throws IOException {

        // Pass 1: count rows so the split works if the dataset changes.
        long totalRows = countRows();

        long trainSize = (long) (totalRows * 0.70);
        long validationSize = (long) (totalRows * 0.15);
        long testSize = totalRows - trainSize - validationSize;

        SplitStats train =
                new SplitStats("TRAIN", 0, trainSize);

        SplitStats validation =
                new SplitStats("VALIDATION", trainSize, validationSize);

        SplitStats test =
                new SplitStats(
                        "TEST",
                        trainSize + validationSize,
                        testSize
                );

        // Pass 2: accumulate statistics for each chronological period.
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
            long rowIndex = 0;

            for (CSVRecord row : parser) {

                int label =
                        Integer.parseInt(row.get("isFraud"));

                long time =
                        Long.parseLong(row.get("TransactionDT"));

                if (rowIndex < train.endExclusive()) {
                    train.add(time, label);

                } else if (rowIndex < validation.endExclusive()) {
                    validation.add(time, label);

                } else {
                    test.add(time, label);
                }

                rowIndex++;
            }

            if (rowIndex != totalRows) {
                throw new IOException(
                        "Row count changed between passes."
                );
            }
        }

        System.out.println("=== FINFORGE TEMPORAL SPLIT AUDIT ===");
        System.out.println("Total transactions: " + totalRows);
        System.out.println("Split method: chronological row order");

        train.print();
        validation.print();
        test.print();

        System.out.println("\nSplit boundaries");
        System.out.println("Training ends at row index: "
                + train.endExclusive());
        System.out.println("Validation ends at row index: "
                + validation.endExclusive());

        Path output = Path.of(
                "reports", "temporal_split_audit.txt"
        );

        Files.createDirectories(output.getParent());

        String report = train.asText()
                + "\n" + validation.asText()
                + "\n" + test.asText();

        Files.writeString(output, report, StandardCharsets.UTF_8);

        System.out.println("\nReport saved to: "
                + output.toAbsolutePath());
    }

    private static long countRows() throws IOException {

        long count = 0;

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
            for (CSVRecord ignored : parser) {
                count++;
            }
        }

        return count;
    }

    private static class SplitStats {

        final String name;
        final long startIndex;
        final long size;

        long rows = 0;
        long fraud = 0;
        long minTime = Long.MAX_VALUE;
        long maxTime = Long.MIN_VALUE;

        SplitStats(String name, long startIndex, long size) {
            this.name = name;
            this.startIndex = startIndex;
            this.size = size;
        }

        long endExclusive() {
            return startIndex + size;
        }

        void add(long time, int label) {
            rows++;

            if (label == 1) {
                fraud++;
            }

            minTime = Math.min(minTime, time);
            maxTime = Math.max(maxTime, time);
        }

        double fraudRate() {
            return rows == 0 ? 0 : 100.0 * fraud / rows;
        }

        String asText() {
            return String.format(
                    Locale.US,
                    "%s | Rows: %d | Fraud: %d | "
                            + "Fraud rate: %.2f%% | "
                            + "TransactionDT range: %d to %d%n",
                    name,
                    rows,
                    fraud,
                    fraudRate(),
                    minTime,
                    maxTime
            );
        }

        void print() {
            System.out.print(asText());
        }
    }
}
