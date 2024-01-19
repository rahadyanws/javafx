package com.tensorflow.javafx;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.HBox;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.File;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class ImageClassificationAppBeta extends Application {

    private ImageView imageView;
    private Button classifyButton;

    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("Image Classification with TensorFlow");

        // Create ImageView
        imageView = new ImageView();
        imageView.setFitWidth(400);
        imageView.setFitHeight(300);

        // Create Button to open FileChooser
        Button openButton = new Button("Open Image");
        openButton.setOnAction(e -> openImage());

        // Create Button to perform classification
        classifyButton = new Button("Classify");
        classifyButton.setDisable(true);
        classifyButton.setOnAction(e -> classifyImage());

        // Create HBox layout and add the ImageView and Buttons to it
        HBox hbox = new HBox(10); // 10 is the spacing between nodes
        hbox.getChildren().addAll(imageView, openButton, classifyButton);

        // Create a Scene with the HBox as the root and set its size
        Scene scene = new Scene(hbox, 600, 400);

        // Set the Scene to the Stage
        primaryStage.setScene(scene);

        // Show the Stage
        primaryStage.show();
    }

    private void openImage() {
        // Create a FileChooser
        FileChooser fileChooser = new FileChooser();
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("Image Files", "*.png", "*.jpg", "*.gif"));

        // Show the FileChooser dialog
        File selectedFile = fileChooser.showOpenDialog(null);

        // Load the selected image into the ImageView
        if (selectedFile != null) {
            Image selectedImage = new Image(selectedFile.toURI().toString());
            imageView.setImage(selectedImage);
            classifyButton.setDisable(false);
        }
    }

    private void classifyImage() {
        // Load TensorFlow model
        try (Graph graph = new Graph();
             Session session = new Session(graph)) {

            // Load the pre-trained TensorFlow model
                byte[] graphDef = readAllBytes(Paths.get("file:///D:/Bootcamp/frozen_model/save_model.pb")); // Replace with the actual path to your model
            graph.importGraphDef(graphDef);

            // Preprocess the image and convert it to a tensor
            Image image = imageView.getImage();
            float[] inputValues = preprocessImage(image);
            try (Tensor<Float> inputTensor = Tensor.create(inputValues, Float.class)) {

                // Perform inference
                List<Tensor<?>> outputTensors = session.runner()
                        .feed("input_tensor_name", inputTensor)  // Replace with the actual input tensor name
                        .fetch("output_tensor_name")            // Replace with the actual output tensor name
                        .run();

                // Process the output tensor (classification results)
                Tensor<?> outputTensor = outputTensors.get(0);
                float[] outputValues = new float[(int) outputTensor.shape()[1]];
                FloatBuffer floatBuffer = FloatBuffer.wrap(outputValues);
                outputTensor.writeTo(floatBuffer);

                // Display or use the classification results as needed
                System.out.println("Classification Results:");
                for (int i = 0; i < outputValues.length; i++) {
                    System.out.println("Class " + i + ": " + outputValues[i]);
                }

                // Release the output tensor resources
                outputTensor.close();
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private float[] preprocessImage(Image image) {
        // Implement image preprocessing logic based on your model requirements
        // Convert the JavaFX Image to a float array suitable for input to the model
        // You may need to resize, normalize, and format the image accordingly
        // For simplicity, this example assumes a flat array of pixel values for demonstration purposes
        int width = (int) image.getWidth();
        int height = (int) image.getHeight();

        float[] pixelValues = new float[width * height * 3]; // Assuming RGB image
        // Placeholder: populate pixelValues with actual pixel data from the image

        return pixelValues;
    }

    private static byte[] readAllBytes(Path path) throws Exception {
        return Files.readAllBytes(path);
    }

    public static void main(String[] args) {
        // Make sure to set the TensorFlow native library path
        System.setProperty("org.tensorflow.NativeLibrary", "/path/to/native/libraries");

        launch(args);
    }
}

