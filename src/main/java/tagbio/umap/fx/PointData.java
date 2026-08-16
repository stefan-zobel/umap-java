package tagbio.umap.fx;

/**
 * A numeric table with a name and a class for every row, which is everything the viewer needs
 * of its input.
 *
 * <p>Two things implement this: {@link LabelledData} for delimited text and {@link MnistData}
 * for the MNIST IDX pair. Keeping them behind one interface is what lets the drawing, the
 * legend and the projection stay unaware of where their points came from.
 */
interface PointData {

  /**
   * The attribute values, one row per sample.
   * @return the data rows
   */
  float[][] getData();

  /**
   * The name of each row, shown in the hover readout.
   * @return the sample names
   */
  String[] getSampleNames();

  /**
   * The distinct class names, indexed by the values of {@link #getClassIndices()}.
   * @return the class names
   */
  String[] getClassNames();

  /**
   * The class of each row, as an index into {@link #getClassNames()}.
   * @return one class index per row
   */
  int[] getClassIndices();
}
