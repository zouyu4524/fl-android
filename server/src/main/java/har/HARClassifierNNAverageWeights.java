package har;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class HARClassifierNNAverageWeights {

    private static final String updatedModel = "res/model/trained_har_nn_updated.zip";
    private static final String originModel = "C:\\Users\\yuze\\Dropbox\\dl4j_codes\\Trained_HAR_NN.zip";
    private static final String onDeviceModelPath = "C:\\Users\\yuze\\Desktop\\on-device-model";

    public static void AverageWeights(List<File> files, File originModel, int layer, double alpha) {
        /*
            files indicates locations that mobile device uploaded model
            originModel is the model maintained by the server
            layerName is the layer to be averaged
            alpha is a coefficient indicates the weight of original model for the updated model
            currently, we just do transfer learning on the devices and we assume that it happens only at
            the last layer (i.e., the output layer) and keep other layers friezed. Therefore, we just need
            to average weights over the last layer.
         */
        // load original model
        MultiLayerNetwork model = null;
        try {
            model = ModelSerializer.restoreMultiLayerNetwork(originModel, false);
        } catch (IOException e) {
            e.printStackTrace();
        }
        Map<String, INDArray> paramTable = model.paramTable();
        INDArray weight = paramTable.get(String.format("%d_W", layer));
        INDArray bias = paramTable.get(String.format("%d_b", layer));
        INDArray avgWeights = weight.mul(alpha);
        INDArray avgBias = bias.mul(alpha);

        // average weights over mobile devices' models
        int len = files.size();
        for (int i = 0; i < len; i++) {
            try {
                model = ModelSerializer.restoreMultiLayerNetwork(files.get(i), false);
            } catch (IOException e) {
                e.printStackTrace();
            }
            paramTable = model.paramTable();
            weight = paramTable.get(String.format("%d_W", layer));
            avgWeights = avgWeights.add(weight.mul(1.0-alpha).div(len));
            bias = paramTable.get(String.format("%d_b", layer));
            avgBias = avgBias.add(bias.mul(1.0-alpha).div(len));
        }
        model.setParam(String.format("%d_W", layer), avgWeights);
        model.setParam(String.format("%d_b", layer), avgBias);
        try {
            ModelSerializer.writeModel(model, updatedModel, false);
        } catch (IOException e){
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws Exception {

        File dir = new File(onDeviceModelPath);
        File[] listOfFiles = dir.listFiles();
        List<File> models = new ArrayList<>();
        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile()) {
                models.add(listOfFiles[i]);
            }
        }
        File originalModel = new File(originModel);
        AverageWeights(models, originalModel, 2, 0.0);
    }
}
