import java.io.*;

public class Utils {
    private static int CONCATENATE_VERTICAL = 10;
    private static int CONCATENATE_HORIZONTAL = 20;
    private static final char DSV_QUOTE = '"';
    private static final char DSV_LF = '\n';
    private static final char DSV_CR = '\r';
    private static final String DSV_QUOTE_AS_STRING = String.valueOf('"');

    public static void concatenate(int concatMode, Object matrixA, Object matrixB) {

    }

    public static void saveToFile(ProteinsDataset dataset) throws IOException {
        ObjectOutputStream out = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(new File(
                "C:\\Users\\CECSAdmin\\OneDrive - University of Tennessee at Chattanooga" +
                        "\\Projects\\ProteinGraph\\exports\\data"))));
        out.writeObject(dataset);
        out.close();
    }

    public static ProteinsDataset readFile() throws IOException, ClassNotFoundException {
        ObjectInputStream in= new ObjectInputStream(new BufferedInputStream(new FileInputStream(new File(
                "C:\\Users\\CECSAdmin\\OneDrive - University of Tennessee at Chattanooga" +
                        "\\Projects\\ProteinGraph\\exports\\data"))));
        Object p = in.readObject();
        in.close();

        return (ProteinsDataset) p;
    }

    public static String escapeDSV(String input, char delimiter) {
        char[] specialChars = new char[]{delimiter, '"', '\n', '\r'};
        boolean containsSpecial = false;

        for (int i = 0; i < specialChars.length; ++i) {
            if (input.contains(String.valueOf(specialChars[i]))) {
                containsSpecial = true;
                break;
            }
        }

        if (containsSpecial) {
            String var10000 = DSV_QUOTE_AS_STRING;
            return var10000 + input.replaceAll(DSV_QUOTE_AS_STRING, DSV_QUOTE_AS_STRING + DSV_QUOTE_AS_STRING) + DSV_QUOTE_AS_STRING;
        } else {
            return input;
        }
    }
}
