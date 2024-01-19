module com.tensorflow.javafx {
    requires javafx.controls;
    requires javafx.fxml;
    requires libtensorflow;


    opens com.tensorflow.javafx to javafx.fxml;
    exports com.tensorflow.javafx;
}