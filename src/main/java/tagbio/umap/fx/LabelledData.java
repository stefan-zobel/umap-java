package tagbio.umap.fx;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.LineNumberReader;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPInputStream;

/**
 * A numeric table read from a delimited text file, where the first column holds a
 * <code>class:id</code> sample name and the remaining columns are numeric attributes.
 * This is the layout of the iris and digits files under <code>src/test/resources</code>,
 * so those can be projected without conversion.
 */
public final class LabelledData implements PointData {

  private final float[][] mData;
  private final String[] mSampleNames;
  private final String[] mClassNames;
  private final int[] mClassIndices;

  private LabelledData(final float[][] data, final String[] sampleNames, final String[] classNames, final int[] classIndices) {
    mData = data;
    mSampleNames = sampleNames;
    mClassNames = classNames;
    mClassIndices = classIndices;
  }

  /**
   * Read a delimited data file. Comma separated when the name contains <code>.csv</code>
   * and tab separated otherwise; <code>.gz</code> files are decompressed while reading.
   * Rows whose field count disagrees with the header are skipped with a warning, matching
   * the behaviour of the test data loader.
   * @param file path to the data file
   * @return the parsed contents
   * @throws IOException if the file cannot be read, has no header, or holds no usable rows
   */
  public static LabelledData load(final String file) throws IOException {
    final String delimiter = file.contains(".csv") ? "," : "\t";
    final List<float[]> rows = new ArrayList<>();
    final List<String> sampleNames = new ArrayList<>();
    final List<Integer> classes = new ArrayList<>();
    // Insertion ordered so that class indices, and therefore colours, follow the order
    // in which the classes first appear in the file.
    final Map<String, Integer> classIndex = new LinkedHashMap<>();

    try (final LineNumberReader reader = new LineNumberReader(new InputStreamReader(open(file)))) {
      final String header = reader.readLine();
      if (header == null) {
        throw new IOException("No header line in " + file);
      }
      final int attributeCount = header.trim().split(delimiter).length - 1;
      if (attributeCount < 2) {
        throw new IOException("Expected a sample column and at least two attribute columns in " + file);
      }
      String line = reader.readLine();
      while (line != null) {
        final String[] parts = line.trim().split(delimiter);
        if (parts.length != attributeCount + 1) {
          System.err.println("Incorrect number of fields in: " + line + " ...skipping");
        } else {
          final float[] values = parseValues(parts, attributeCount);
          if (values == null) {
            System.err.println("Unparseable number in: " + line + " ...skipping");
          } else {
            rows.add(values);
            sampleNames.add(parts[0]);
            final String className = parts[0].split(":")[0];
            classIndex.putIfAbsent(className, classIndex.size());
            classes.add(classIndex.get(className));
          }
        }
        line = reader.readLine();
      }
    }

    if (rows.isEmpty()) {
      throw new IOException("No usable data rows in " + file);
    }
    final int[] classIndices = new int[classes.size()];
    for (int i = 0; i < classIndices.length; ++i) {
      classIndices[i] = classes.get(i);
    }
    return new LabelledData(rows.toArray(new float[0][]), sampleNames.toArray(new String[0]),
      classIndex.keySet().toArray(new String[0]), classIndices);
  }

  /**
   * Parse the attribute columns of one row. A single unparseable cell discards the whole
   * row rather than the whole file, matching how rows with the wrong field count are
   * treated.
   * @param parts the fields of the row, the first being the sample name
   * @param attributeCount how many attribute columns to read
   * @return the values, or null if any of them is not a number
   */
  private static float[] parseValues(final String[] parts, final int attributeCount) {
    final float[] values = new float[attributeCount];
    for (int k = 0; k < attributeCount; ++k) {
      try {
        values[k] = Float.parseFloat(parts[k + 1]);
      } catch (final NumberFormatException e) {
        return null;
      }
    }
    return values;
  }

  private static InputStream open(final String file) throws IOException {
    final InputStream stream = new BufferedInputStream(new FileInputStream(file));
    return file.endsWith(".gz") ? new GZIPInputStream(stream) : stream;
  }

  /**
   * The attribute values, one row per sample.
   * @return the data rows
   */
  @Override
  public float[][] getData() {
    return mData;
  }

  /**
   * The full sample name of each row, including the <code>:id</code> suffix.
   * @return the sample names
   */
  @Override
  public String[] getSampleNames() {
    return mSampleNames;
  }

  /**
   * The distinct class names, in order of first appearance in the file.
   * @return the class names
   */
  @Override
  public String[] getClassNames() {
    return mClassNames;
  }

  /**
   * The class of each row, as an index into {@link #getClassNames()}.
   * @return one class index per row
   */
  @Override
  public int[] getClassIndices() {
    return mClassIndices;
  }
}
