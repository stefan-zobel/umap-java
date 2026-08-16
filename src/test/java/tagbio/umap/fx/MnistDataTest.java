package tagbio.umap.fx;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.zip.GZIPOutputStream;

import junit.framework.TestCase;

/**
 * The reader is exercised against IDX files written here rather than against the real MNIST
 * files, so that the suite runs whether or not those have been copied into this project.
 */
public class MnistDataTest extends TestCase {

  private static final int IMAGE_MAGIC = 2051;
  private static final int LABEL_MAGIC = 2049;
  private static final int EDGE = 28;
  private static final int PIXELS = EDGE * EDGE;

  private File mRoot;

  @Override
  protected void setUp() throws IOException {
    mRoot = Files.createTempDirectory("mnist-data-test").toFile();
  }

  @Override
  protected void tearDown() {
    delete(mRoot);
    mRoot = null;
  }

  private static void delete(final File file) {
    if (file == null) {
      return;
    }
    final File[] children = file.listFiles();
    if (children != null) {
      for (final File child : children) {
        delete(child);
      }
    }
    if (!file.delete()) {
      file.deleteOnExit();
    }
  }

  private File dataset(final int images) throws IOException {
    return dataset(images, images, IMAGE_MAGIC, LABEL_MAGIC, EDGE);
  }

  /**
   * Writes a pair of IDX files. Pixel <code>p</code> of image <code>i</code> is
   * <code>(i * 7 + p) % 256</code>, which is not symmetric under transposition, so a reader
   * that forgot to transpose would not pass.
   */
  private File dataset(final int images, final int labels, final int imageMagic,
    final int labelMagic, final int edge) throws IOException {
    final File directory = new File(mRoot, "d" + mRoot.list().length);
    assertTrue(directory.mkdir());
    try (DataOutputStream out = gzip(new File(directory, "train-images-idx3-ubyte.gz"))) {
      out.writeInt(imageMagic);
      out.writeInt(images);
      out.writeInt(edge);
      out.writeInt(edge);
      for (int i = 0; i < images; ++i) {
        for (int p = 0; p < edge * edge; ++p) {
          out.write((i * 7 + p) % 256);
        }
      }
    }
    try (DataOutputStream out = gzip(new File(directory, "train-labels-idx1-ubyte.gz"))) {
      out.writeInt(labelMagic);
      out.writeInt(labels);
      for (int i = 0; i < labels; ++i) {
        out.write(i % 10);
      }
    }
    return directory;
  }

  private static DataOutputStream gzip(final File file) throws IOException {
    return new DataOutputStream(new GZIPOutputStream(new FileOutputStream(file)));
  }

  private static int rawPixel(final int image, final int pixel) {
    return (image * 7 + pixel) % 256;
  }

  public void testIsMnistDirectoryRecognisesThePair() throws IOException {
    assertTrue(MnistData.isMnistDirectory(dataset(3)));
  }

  public void testIsMnistDirectoryRejectsAnythingElse() throws IOException {
    assertFalse(MnistData.isMnistDirectory(new File(mRoot, "nowhere")));
    final File empty = new File(mRoot, "empty");
    assertTrue(empty.mkdir());
    assertFalse(MnistData.isMnistDirectory(empty));
    // A delimited data file is the other thing the viewer is launched with.
    final File file = new File(mRoot, "data.tsv");
    assertTrue(file.createNewFile());
    assertFalse(MnistData.isMnistDirectory(file));
  }

  public void testAvailableReportsTheRecordCount() throws IOException {
    assertEquals(37, MnistData.available(dataset(37)));
  }

  public void testLoadReadsAPrefixRatherThanTheWholeFile() throws IOException {
    final MnistData data = MnistData.load(dataset(50), 12);
    assertEquals(12, data.getData().length);
    assertEquals(12, data.getSampleNames().length);
    assertEquals(12, data.getClassIndices().length);
    assertEquals(MnistData.PIXELS, data.getData()[0].length);
  }

  /**
   * The encoding is transposed within the image and inverted so that the background is 1.0.
   * Both halves matter, so both are asserted.
   */
  public void testPixelsAreTransposedAndInverted() throws IOException {
    final MnistData data = MnistData.load(dataset(5), 5);
    for (int image = 0; image < 5; ++image) {
      for (int p = 0; p < PIXELS; ++p) {
        final int transposed = (p % EDGE) * EDGE + (p / EDGE);
        assertEquals("image " + image + " pixel " + p,
          rawPixel(image, transposed) != 0 ? 0.0f : 1.0f, data.getData()[image][p], 0.0f);
      }
    }
  }

  public void testEveryValueIsZeroOrOne() throws IOException {
    final MnistData data = MnistData.load(dataset(4), 4);
    for (final float[] row : data.getData()) {
      for (final float value : row) {
        assertTrue("unexpected " + value, value == 0.0f || value == 1.0f);
      }
    }
  }

  public void testClassIndexIsTheDigitItself() throws IOException {
    final MnistData data = MnistData.load(dataset(25), 25);
    for (int i = 0; i < 25; ++i) {
      assertEquals(i % 10, data.getClassIndices()[i]);
    }
  }

  /**
   * Colours follow the class index, so a digit has to keep its index no matter how many points
   * are loaded. An order of first appearance would repaint the whole plot whenever the point
   * count changed.
   */
  public void testClassIndicesDoNotDependOnThePointCount() throws IOException {
    final File directory = dataset(60);
    final MnistData few = MnistData.load(directory, 12);
    final MnistData many = MnistData.load(directory, 60);
    for (int i = 0; i < 12; ++i) {
      assertEquals(few.getClassIndices()[i], many.getClassIndices()[i]);
    }
    assertEquals(few.getClassNames().length, many.getClassNames().length);
  }

  public void testThereAreAlwaysTenClasses() throws IOException {
    final MnistData data = MnistData.load(dataset(3), 3);
    assertEquals(10, data.getClassNames().length);
    for (int digit = 0; digit < 10; ++digit) {
      assertEquals(String.valueOf(digit), data.getClassNames()[digit]);
    }
  }

  public void testSampleNamesCarryTheDigitAndTheRow() throws IOException {
    final MnistData data = MnistData.load(dataset(13), 13);
    assertEquals("0:0", data.getSampleNames()[0]);
    assertEquals("7:7", data.getSampleNames()[7]);
    assertEquals("2:12", data.getSampleNames()[12]);
  }

  public void testAskingForMoreThanTheFileHoldsFails() throws IOException {
    final File directory = dataset(4);
    try {
      MnistData.load(directory, 5);
      fail("expected an IOException");
    } catch (final IOException e) {
      assertTrue(String.valueOf(e.getMessage()), String.valueOf(e.getMessage()).contains("only 4"));
    }
  }

  public void testAMissingFileFails() {
    try {
      MnistData.available(new File(mRoot, "nowhere"));
      fail("expected an IOException");
    } catch (final IOException e) {
      assertTrue(String.valueOf(e.getMessage()), String.valueOf(e.getMessage()).contains("No such file"));
    }
  }

  public void testAWrongMagicNumberFails() throws IOException {
    final File directory = dataset(4, 4, 1234, LABEL_MAGIC, EDGE);
    try {
      MnistData.available(directory);
      fail("expected an IOException");
    } catch (final IOException e) {
      assertTrue(String.valueOf(e.getMessage()), String.valueOf(e.getMessage()).contains("magic"));
    }
  }

  public void testAnUnexpectedImageSizeFails() throws IOException {
    final File directory = dataset(4, 4, IMAGE_MAGIC, LABEL_MAGIC, 20);
    try {
      MnistData.load(directory, 4);
      fail("expected an IOException");
    } catch (final IOException e) {
      assertTrue(String.valueOf(e.getMessage()), String.valueOf(e.getMessage()).contains("20 x 20"));
    }
  }

  public void testFewerLabelsThanImagesFails() throws IOException {
    final File directory = dataset(10, 3, IMAGE_MAGIC, LABEL_MAGIC, EDGE);
    try {
      MnistData.load(directory, 10);
      fail("expected an IOException");
    } catch (final IOException e) {
      assertTrue(String.valueOf(e.getMessage()), String.valueOf(e.getMessage()).contains("only 3"));
    }
  }

  public void testANonPositiveCountFails() throws IOException {
    final File directory = dataset(4);
    try {
      MnistData.load(directory, 0);
      fail("expected an IOException");
    } catch (final IOException e) {
      assertTrue(String.valueOf(e.getMessage()), String.valueOf(e.getMessage()).contains("0 images"));
    }
  }
}
