import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;

public class GCN {
    private void loadInput() {
    }

    private INDArray preprocess(int type, INDArray adjMatrix, INDArray featureMatrix) {
        // types include kipf multiplication
        switch (type) {
            case 0://kipf style

                adjMatrix.mmul(featureMatrix);

                break;
            default:
        }
        return null;
    }
}
