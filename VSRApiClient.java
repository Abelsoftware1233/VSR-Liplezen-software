package com.vsr.app;

import org.json.JSONObject;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.UUID;

/**
 * Client for the local VSR REST API (api/server.py), running on
 * 127.0.0.1:6040. This class only ever talks to localhost — there is no
 * external network call anywhere in this file.
 */
public class VSRApiClient {

    public static final String API_BASE = "http://127.0.0.1:6040";
    private static final String BOUNDARY = "----VSRBoundary" + UUID.randomUUID();

    private final HttpClient httpClient;

    public VSRApiClient() {
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .build();
    }

    /** Simple result holder for a successful transcription. */
    public record TranscriptionResult(
            String primaryTranscript,
            String secondaryTranscript,
            String finalTranscript,
            int framesTotal,
            int framesWithFaceDetected
    ) {}

    /** Thrown when the API is unreachable or returns an error. */
    public static class VSRApiException extends Exception {
        public VSRApiException(String message) {
            super(message);
        }
        public VSRApiException(String message, Throwable cause) {
            super(message, cause);
        }
    }

    /** Checks whether the local API is up before we let the user try to upload anything. */
    public boolean checkHealth() {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(API_BASE + "/health"))
                    .timeout(Duration.ofSeconds(3))
                    .GET()
                    .build();
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            return response.statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Uploads a video file to the local API and returns the transcription result.
     *
     * @param videoFile    path to the video file on disk
     * @param useSecondary whether to also run the AV-HuBERT ensemble model
     */
    public TranscriptionResult transcribe(Path videoFile, boolean useSecondary) throws VSRApiException {
        try {
            byte[] videoBytes = Files.readAllBytes(videoFile);
            String fileName = videoFile.getFileName().toString();

            byte[] multipartBody = buildMultipartBody(videoBytes, fileName);

            URI uri = URI.create(API_BASE + "/transcribe?use_secondary=" + useSecondary);
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(uri)
                    .timeout(Duration.ofMinutes(5)) // CPU inference can be slow
                    .header("Content-Type", "multipart/form-data; boundary=" + BOUNDARY)
                    .POST(HttpRequest.BodyPublishers.ofByteArray(multipartBody))
                    .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() != 200) {
                String detail = tryExtractDetail(response.body());
                throw new VSRApiException("API error (" + response.statusCode() + "): " + detail);
            }

            JSONObject json = new JSONObject(response.body());
            return new TranscriptionResult(
                    json.optString("primary_transcript", ""),
                    json.isNull("secondary_transcript") ? null : json.optString("secondary_transcript", null),
                    json.optString("final_transcript", ""),
                    json.optInt("frames_total", 0),
                    json.optInt("frames_with_face_detected", 0)
            );

        } catch (IOException | InterruptedException e) {
            throw new VSRApiException(
                    "Could not reach the local API on port 6040. Make sure it is running: " +
                    "uvicorn api.server:app --host 127.0.0.1 --port 6040", e);
        }
    }

    private String tryExtractDetail(String body) {
        try {
            return new JSONObject(body).optString("detail", body);
        } catch (Exception e) {
            return body;
        }
    }

    private byte[] buildMultipartBody(byte[] videoBytes, String fileName) throws IOException {
        String header =
                "--" + BOUNDARY + "\r\n" +
                "Content-Disposition: form-data; name=\"file\"; filename=\"" + fileName + "\"\r\n" +
                "Content-Type: video/mp4\r\n\r\n";
        String footer = "\r\n--" + BOUNDARY + "--\r\n";

        var out = new java.io.ByteArrayOutputStream();
        out.write(header.getBytes());
        out.write(videoBytes);
        out.write(footer.getBytes());
        return out.toByteArray();
    }
}
