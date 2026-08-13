package tagbio.umap.fx;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.zip.GZIPOutputStream;

import junit.framework.TestCase;

public class LabelledDataTest extends TestCase {

  private final List<File> mTemporary = new ArrayList<>();

  @Override
  protected void tearDown() {
    for (final File file : mTemporary) {
      if (!file.delete()) {
        file.deleteOnExit();
      }
    }
    mTemporary.clear();
  }

  private String write(final String suffix, final String... lines) throws IOException {
    final File file = File.createTempFile("labelled-data-test", suffix);
    mTemporary.add(file);
    try (final Writer writer = new OutputStreamWriter(stream(file, suffix), StandardCharsets.UTF_8)) {
      for (final String line : lines) {
        writer.write(line);
        writer.write('\n');
      }
    }
    return file.getAbsolutePath();
  }

  private static OutputStream stream(final File file, final String suffix) throws IOException {
    final OutputStream out = new FileOutputStream(file);
    return suffix.endsWith(".gz") ? new GZIPOutputStream(out) : out;
  }

  public void testReadsTabSeparated() throws IOException {
    final String path = write(".tsv",
      "sample\tatt0\tatt1",
      "setosa:0\t5.1\t3.5",
      "setosa:1\t4.9\t3.0",
      "virginica:2\t6.3\t3.3");
    final LabelledData data = LabelledData.load(path);
    assertEquals(3, data.getData().length);
    assertEquals(2, data.getData()[0].length);
    assertEquals(5.1f, data.getData()[0][0], 1.0e-6f);
    assertEquals(3.3f, data.getData()[2][1], 1.0e-6f);
  }

  public void testReadsCommaSeparated() throws IOException {
    final String path = write(".csv",
      "sample,att0,att1",
      "a:0,1.5,2.5",
      "b:1,3.5,4.5");
    final LabelledData data = LabelledData.load(path);
    assertEquals(2, data.getData().length);
    assertEquals(1.5f, data.getData()[0][0], 1.0e-6f);
    assertEquals(4.5f, data.getData()[1][1], 1.0e-6f);
  }

  public void testReadsGzipped() throws IOException {
    final String path = write(".tsv.gz",
      "sample\tatt0\tatt1",
      "a:0\t1\t2",
      "a:1\t3\t4");
    final LabelledData data = LabelledData.load(path);
    assertEquals(2, data.getData().length);
    assertEquals(3.0f, data.getData()[1][0], 1.0e-6f);
  }

  public void testClassNamesComeFromThePrefixInFirstAppearanceOrder() throws IOException {
    final String path = write(".tsv",
      "sample\tatt0\tatt1",
      "virginica:0\t1\t2",
      "setosa:1\t3\t4",
      "virginica:2\t5\t6",
      "versicolor:3\t7\t8");
    final LabelledData data = LabelledData.load(path);
    assertTrue(Arrays.toString(data.getClassNames()),
      Arrays.equals(new String[] {"virginica", "setosa", "versicolor"}, data.getClassNames()));
    assertTrue(Arrays.toString(data.getClassIndices()),
      Arrays.equals(new int[] {0, 1, 0, 2}, data.getClassIndices()));
  }

  public void testSampleNamesKeepTheirSuffix() throws IOException {
    final String path = write(".tsv",
      "sample\tatt0\tatt1",
      "setosa:7\t1\t2");
    final LabelledData data = LabelledData.load(path);
    assertEquals("setosa:7", data.getSampleNames()[0]);
  }

  public void testSampleNameWithoutColonIsItsOwnClass() throws IOException {
    final String path = write(".tsv",
      "sample\tatt0\tatt1",
      "plain\t1\t2");
    final LabelledData data = LabelledData.load(path);
    assertEquals(1, data.getClassNames().length);
    assertEquals("plain", data.getClassNames()[0]);
  }

  public void testRowWithWrongFieldCountIsSkipped() throws IOException {
    final String path = write(".tsv",
      "sample\tatt0\tatt1",
      "a:0\t1\t2",
      "a:1\t3",
      "a:2\t4\t5");
    final LabelledData data = LabelledData.load(path);
    assertEquals(2, data.getData().length);
    assertEquals(4.0f, data.getData()[1][0], 1.0e-6f);
  }

  public void testRowWithUnparseableNumberIsSkipped() throws IOException {
    final String path = write(".tsv",
      "sample\tatt0\tatt1",
      "a:0\t1\t2",
      "a:1\tnot-a-number\t3",
      "a:2\t4\t5");
    final LabelledData data = LabelledData.load(path);
    assertEquals(2, data.getData().length);
    // The skipped row must not leave a stray label behind.
    assertEquals(2, data.getSampleNames().length);
    assertEquals(2, data.getClassIndices().length);
    assertEquals("a:2", data.getSampleNames()[1]);
  }

  public void testMissingFileFails() {
    try {
      LabelledData.load("no-such-file-here.tsv");
      fail("expected an IOException");
    } catch (final IOException e) {
      // expected
    }
  }

  public void testEmptyFileFails() throws IOException {
    final String path = write(".tsv");
    try {
      LabelledData.load(path);
      fail("expected an IOException");
    } catch (final IOException e) {
      assertTrue(e.getMessage(), e.getMessage().contains("No header line"));
    }
  }

  public void testTooFewAttributeColumnsFails() throws IOException {
    final String path = write(".tsv",
      "sample\tatt0",
      "a:0\t1");
    try {
      LabelledData.load(path);
      fail("expected an IOException");
    } catch (final IOException e) {
      assertTrue(e.getMessage(), e.getMessage().contains("attribute columns"));
    }
  }

  public void testHeaderOnlyFails() throws IOException {
    final String path = write(".tsv", "sample\tatt0\tatt1");
    try {
      LabelledData.load(path);
      fail("expected an IOException");
    } catch (final IOException e) {
      assertTrue(e.getMessage(), e.getMessage().contains("No usable data rows"));
    }
  }

  public void testAllRowsUnusableFails() throws IOException {
    final String path = write(".tsv",
      "sample\tatt0\tatt1",
      "a:0\tx\ty");
    try {
      LabelledData.load(path);
      fail("expected an IOException");
    } catch (final IOException e) {
      assertTrue(e.getMessage(), e.getMessage().contains("No usable data rows"));
    }
  }
}
