import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.jgrapht.graph.DefaultWeightedEdge;

import java.util.Map;
import java.util.Objects;

@NoArgsConstructor
@Data
@AllArgsConstructor
public class Bond extends DefaultWeightedEdge {
    public static final String PEPTIDE_BOND = "PB";
    public static final String HYDROGEN_BOND = "HB";
    String label;// either hydrogen Bond or peptide bond
    String bondID;

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Bond)) return false;
        Bond bond = (Bond) o;
        return getLabel().equals(bond.getLabel()) &&
                getBondID().equals(bond.getBondID());
    }

    @Override
    public int hashCode() {
        return Objects.hash(getLabel(), getBondID());
    }
}
