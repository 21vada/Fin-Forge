
package com.finforge;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

public class csvtest {

    public static void main(String[] args) throws IOException {

        // Locate the dataset in the parent FinForge folder.
        Path csvPath = Path.of(
                "C:/Users/pc/Documents/fin forge/ieee-fraud-detection/train_transaction.csv"
        );

        if (!Files.exists(csvPath)) {
            throw new FileNotFoundException(
                    "Dataset not found: "
                            + csvPath.toAbsolutePath()
            );
        }

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
            System.out.println("=== FINFORGE CSV READER ===");
            System.out.println("Dataset: " + csvPath.toAbsolutePath());
            System.out.println(
                    "Number of columns: " + parser.getHeaderMap().size()
            );

            int rowsRead = 0;

            for (CSVRecord row : parser) {

                if (rowsRead < 5) {
                    System.out.printf(
                            "ID: %s | Amount: %s | Product: %s | Fraud: %s%n",
                            row.get("TransactionID"),
                            row.get("TransactionAmt"),
                            row.get("ProductCD"),
                            row.get("isFraud")
                    );
                }

                rowsRead++;

                if (rowsRead >= 10_000) {
                    break;
                }
            }

            System.out.println("----------------------------");
            System.out.println("Rows read: " + rowsRead);
            System.out.println("CSV reading successful!");

        }
    }
}
