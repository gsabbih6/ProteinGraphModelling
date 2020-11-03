import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Map;
import java.util.Objects;

@AllArgsConstructor
@NoArgsConstructor
@Data
public class AminoAcid implements Vertex {
    int id; // residue position
    String label;// type of amino-acid
//    Map<String,Object> property; // a list of properties yet to be define. An example can include weight,spacial locations

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof AminoAcid)) return false;
        AminoAcid aminoAcid = (AminoAcid) o;
        return getId() == aminoAcid.getId() &&
                getLabel().equals(aminoAcid.getLabel());
    }

    @Override
    public int hashCode() {
        return Objects.hash(getId(), getLabel());
    }
}
