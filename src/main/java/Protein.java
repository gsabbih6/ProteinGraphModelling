import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

@Data
public class Protein implements Serializable {
    private INDArray adjacencyMatrix;
    private INDArray featureMatrix;
    private INDArray label;

    public Protein(INDArray adjacencyMatrix, INDArray featureMatrix, INDArray label) {
        this.adjacencyMatrix = adjacencyMatrix;
        this.featureMatrix = featureMatrix;
        this.label = label;
    }
}
