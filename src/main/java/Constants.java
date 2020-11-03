import java.util.Hashtable;

public  class Constants {
    public final static String alanine = "A";
    public final static String alanine_color = "#8d10b2";
    public final static String arginine = "R";
    public final static String arginine_color = "#29160d";
    public final static String asparagine = "N";
    public final static String asparagine_color = "#85ff82";
    public final static String aspartic = "D";
    public final static String aspartic_color = "#a04e45";
    public final static String cysteine = "C";
    public final static String cysteine_color = "#c9e149";
    public final static String glutamine = "Q";
    public final static String glutamine_color = "#c73128";
    public final static String glutamic = "E";
    public final static String glutamic_color = "#ac66c2";
    public final static String glycine = "G";
    public final static String glycine_color = "#52c1b6";
    public final static String histidine = "H";
    public final static String histidine_color = "#8e738c";
    public final static String isoleucine = "I";
    public final static String isoleucine_color = "#f87307";
    public final static String leucine = "L";
    public final static String leucine_color = "#6a9156";
    public final static String lysine = "K";
    public final static String lysine_color = "#83a888";
    public final static String methionine = "M";
    public final static String methionine_color = "#e9fb10";
    public final static String phenylalanine = "F";
    public final static String phenylalanine_color = "#272848";
    public final static String proline = "P";
    public final static String proline_color = "#b02a3a";
    public final static String serine = "S";
    public final static String serine_color = "#a6ad75";
    public final static String threonine = "T";
    public final static String threonine_color = "#fc1ea9";
    public final static String tryptophan = "W";
    public final static String tryptophan_color = "#29160d";
    public final static String tyrosine = "Y";
    public final static String tyrosine_color = "#ced65c";
    public final static String valine = "V";
    public final static String valine_color = "#ce065c";
    //    Sometimes it is not possible two differentiate two closely related amino acids, therefore we have the special cases:
    public final static String asparagine_aspartic = "B";
    public final static String asparagine_aspartic_color = "#ab04f7";
    public final static String glutamine_glutamic = "Z";
    public final static String glutamine_glutamic_color = "#bccc15";

    public static Hashtable<String, String> colormap() {
        Hashtable<String, String> colormap = new Hashtable<>();


        colormap.put(arginine, arginine_color);
        colormap.put(alanine, alanine_color);
        colormap.put(asparagine, asparagine_color);
        colormap.put(asparagine_aspartic, asparagine_aspartic_color);
        colormap.put(glutamic, glutamic_color);
        colormap.put(glutamine_glutamic, glutamine_glutamic_color);
        colormap.put(aspartic, aspartic_color);
        colormap.put(cysteine, cysteine_color);
        colormap.put(serine, serine_color);
        colormap.put(glycine, glycine_color);
        colormap.put(histidine, histidine_color);
        colormap.put(methionine, methionine_color);
        colormap.put(isoleucine, isoleucine_color);
        colormap.put(phenylalanine, phenylalanine_color);

        colormap.put(proline, proline_color);
        colormap.put(leucine, leucine_color);
        colormap.put(lysine, lysine_color);
        colormap.put(threonine, threonine_color);
        colormap.put(glutamine, glutamine_color);
        colormap.put(tryptophan, tryptophan_color);
        colormap.put(tyrosine, tyrosine_color);
        colormap.put(valine, valine_color);

        return colormap;
    }

}
