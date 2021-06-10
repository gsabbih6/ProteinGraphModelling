import java.awt.*;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class ProteinsDataset implements Serializable {
    private List<Protein> dataset;
    private static final long seriaVersionUId = 1L;

    public List<Protein> getDataset() {
        return dataset;
    }

    public void setDataset(Protein protein) {
        if (dataset == null) {
            dataset = new ArrayList<>();
        }
        dataset.add(protein);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        dataset.parallelStream().forEach((e) -> {
            sb.append(e.toString());
            sb.append("\n\n");
        });
        return "ProteinsDataset{" +
                "dataset=" + sb.toString() +
                '}';
    }
}
