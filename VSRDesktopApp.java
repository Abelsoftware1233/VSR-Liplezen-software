package com.vsr.app;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import java.io.File;
import java.nio.file.Path;
import java.util.concurrent.CompletableFuture;

/**
 * Desktop UI for the Visual Speech Recognition tool.
 *
 * Talks to the local Python API (api/server.py) on 127.0.0.1:6040 — same
 * backend the web UI uses. Nothing in this app calls out to the internet;
 * it only ever reaches localhost.
 *
 * Start the API first:
 *   uvicorn api.server:app --host 127.0.0.1 --port 6040
 *
 * Then run this app:
 *   mvn javafx:run
 */
public class VSRDesktopApp extends Application {

    private final VSRApiClient apiClient = new VSRApiClient();
    private File selectedFile;

    private Label statusLabel;
    private Label fileLabel;
    private CheckBox useSecondaryCheckbox;
    private Button transcribeButton;
    private ProgressIndicator progressIndicator;
    private Label progressLabel;
    private TextArea resultArea;
    private TextArea rawOutputArea;
    private Button downloadButton;
    private TitledPane rawOutputPaneRef;
    private String lastTranscript = "";

    @Override
    public void start(Stage stage) {
        stage.setTitle("Visual Speech Recognition");

        Label title = new Label("👄 Visual Speech Recognition");
        title.setFont(Font.font("System", FontWeight.BOLD, 20));
        title.setStyle("-fx-text-fill: #e8eaed;");

        Label subtitle = new Label("Upload a silent video. Mouth movement is transcribed to text — no audio is read.");
        subtitle.setWrapText(true);
        subtitle.setStyle("-fx-text-fill: #888;");

        statusLabel = new Label("Checking connection to local API on port 6040...");
        statusLabel.setStyle("-fx-padding: 8px; -fx-background-color: #2a2f3a; -fx-text-fill: #e8eaed; -fx-background-radius: 6px;");

        TitledPane accuracyNote = new TitledPane();
        accuracyNote.setText("⚠️ Accuracy expectations — please read");
        Label accuracyText = new Label(
                "This app uses open-source pretrained models (Auto-AVSR, optionally AV-HuBERT). " +
                "Even the best published result is around 20% word error rate on clean, well-lit, " +
                "front-facing video — roughly 1 in 5 words can be wrong. Poor lighting, side angles, " +
                "facial hair, or fast speech make this worse. English only. " +
                "See docs/ACCURACY_NOTES.md for details."
        );
        accuracyText.setWrapText(true);
        accuracyNote.setContent(accuracyText);
        accuracyNote.setExpanded(false);

        Button chooseFileButton = new Button("Choose video file...");
        fileLabel = new Label("No file selected");
        fileLabel.setStyle("-fx-text-fill: #888;");
        chooseFileButton.setOnAction(e -> chooseFile(stage));

        useSecondaryCheckbox = new CheckBox(
                "Also run AV-HuBERT and combine results (slower, may slightly improve accuracy)");
        useSecondaryCheckbox.setStyle("-fx-text-fill: #e8eaed;");

        transcribeButton = new Button("Transcribe");
        transcribeButton.setDisable(true);
        transcribeButton.setDefaultButton(true);
        transcribeButton.setOnAction(e -> runTranscription());

        progressIndicator = new ProgressIndicator();
        progressIndicator.setMaxSize(20, 20);
        progressIndicator.setVisible(false);
        progressLabel = new Label(
                "Running face tracking, ROI extraction, and transcription... this can take a while on CPU.");
        progressLabel.setWrapText(true);
        progressLabel.setStyle("-fx-text-fill: #888;");
        progressLabel.setVisible(false);

        HBox progressBox = new HBox(10, progressIndicator, progressLabel);
        progressBox.setAlignment(Pos.CENTER_LEFT);

        Label resultHeader = new Label("Transcript");
        resultHeader.setFont(Font.font("System", FontWeight.BOLD, 14));
        resultHeader.setStyle("-fx-text-fill: #e8eaed;");

        resultArea = new TextArea();
        resultArea.setEditable(false);
        resultArea.setWrapText(true);
        resultArea.setPrefRowCount(4);
        resultArea.setVisible(false);
        resultArea.setManaged(false);

        downloadButton = new Button("Download transcript (.txt)");
        downloadButton.setVisible(false);
        downloadButton.setManaged(false);
        downloadButton.setOnAction(e -> downloadTranscript(stage));

        TitledPane rawOutputPane = new TitledPane();
        rawOutputPane.setText("Raw model output (before language model correction)");
        rawOutputArea = new TextArea();
        rawOutputArea.setEditable(false);
        rawOutputArea.setWrapText(true);
        rawOutputArea.setPrefRowCount(3);
        rawOutputPane.setContent(rawOutputArea);
        rawOutputPane.setExpanded(false);
        rawOutputPane.setVisible(false);
        rawOutputPane.setManaged(false);
        this.rawOutputPaneRef = rawOutputPane;

        VBox root = new VBox(14,
                title, subtitle, statusLabel, accuracyNote,
                chooseFileButton, fileLabel, useSecondaryCheckbox,
                transcribeButton, progressBox,
                resultHeader, resultArea, downloadButton, rawOutputPane
        );
        root.setPadding(new Insets(24));
        root.setStyle("-fx-background-color: #0f1115;");

        Scene scene = new Scene(root, 560, 640);
        stage.setScene(scene);
        stage.show();

        checkApiHealthAsync();
    }

    private void chooseFile(Stage stage) {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Select a video file");
        chooser.getExtensionFilters().add(
                new FileChooser.ExtensionFilter("Video files", "*.mp4", "*.mov", "*.avi", "*.mkv"));
        File file = chooser.showOpenDialog(stage);
        if (file != null) {
            selectedFile = file;
            fileLabel.setText(file.getName());
            transcribeButton.setDisable(false);
            resultArea.setVisible(false);
            resultArea.setManaged(false);
            downloadButton.setVisible(false);
            downloadButton.setManaged(false);
        }
    }

    private void checkApiHealthAsync() {
        CompletableFuture.supplyAsync(apiClient::checkHealth).thenAccept(isUp ->
                Platform.runLater(() -> {
                    if (isUp) {
                        statusLabel.setText("Connected to local API on port 6040");
                        statusLabel.setStyle(
                                "-fx-padding: 8px; -fx-background-color: #12261a; -fx-text-fill: #3fb950; -fx-background-radius: 6px;");
                    } else {
                        statusLabel.setText(
                                "Could not reach the local API on port 6040. Start it first with: " +
                                "uvicorn api.server:app --host 127.0.0.1 --port 6040");
                        statusLabel.setStyle(
                                "-fx-padding: 8px; -fx-background-color: #2a1414; -fx-text-fill: #f85149; -fx-background-radius: 6px;");
                    }
                })
        );
    }

    private void runTranscription() {
        if (selectedFile == null) return;

        transcribeButton.setDisable(true);
        progressIndicator.setVisible(true);
        progressLabel.setVisible(true);
        resultArea.setVisible(false);
        resultArea.setManaged(false);

        boolean useSecondary = useSecondaryCheckbox.isSelected();
        Path filePath = selectedFile.toPath();

        CompletableFuture.supplyAsync(() -> {
            try {
                return apiClient.transcribe(filePath, useSecondary);
            } catch (VSRApiClient.VSRApiException e) {
                throw new RuntimeException(e);
            }
        }).whenComplete((result, throwable) -> Platform.runLater(() -> {
            progressIndicator.setVisible(false);
            progressLabel.setVisible(false);
            transcribeButton.setDisable(false);

            if (throwable != null) {
                Alert alert = new Alert(Alert.AlertType.ERROR);
                alert.setTitle("Transcription failed");
                alert.setHeaderText(null);
                alert.setContentText(throwable.getCause() != null
                        ? throwable.getCause().getMessage() : throwable.getMessage());
                alert.showAndWait();
                return;
            }

            lastTranscript = result.finalTranscript();
            resultArea.setText(lastTranscript.isBlank() ? "(empty output)" : lastTranscript);
            resultArea.setVisible(true);
            resultArea.setManaged(true);
            downloadButton.setVisible(true);
            downloadButton.setManaged(true);

            StringBuilder raw = new StringBuilder();
            raw.append("Auto-AVSR: ").append(result.primaryTranscript());
            if (result.secondaryTranscript() != null) {
                raw.append("\nAV-HuBERT: ").append(result.secondaryTranscript());
            }
            rawOutputArea.setText(raw.toString());
            rawOutputPaneRef.setVisible(true);
            rawOutputPaneRef.setManaged(true);
        }));
    }

    private void downloadTranscript(Stage stage) {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Save transcript");
        chooser.setInitialFileName("transcript.txt");
        chooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("Text files", "*.txt"));
        File file = chooser.showSaveDialog(stage);
        if (file != null) {
            try {
                java.nio.file.Files.writeString(file.toPath(), lastTranscript);
            } catch (Exception e) {
                Alert alert = new Alert(Alert.AlertType.ERROR);
                alert.setContentText("Could not save file: " + e.getMessage());
                alert.showAndWait();
            }
        }
    }

    public static void main(String[] args) {
        launch(args);
    }
}
