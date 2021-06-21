import org.tensorflow.SavedModelBundle;
import org.tensorflow.Graph;
import org.tensorflow.Tensor;

public class Main {

    public static void main(String[] args) {
        SavedModelBundle savedModelBundle = SavedModelBundle.load("/Users/henanwan/Documents/workspace/albert-chinese-ner/TensorFlowJavaServer/src/main/resources/model/export/Servo/1591585362", "serve");
        Graph graph = savedModelBundle.graph();
        // printOperations(graph);
//        Tensor result = savedModelBundle.session().runner()
//                .feed("myInput", tensorInput)
//                .fetch("myOutput")
//                .run().get(0);
    }
}