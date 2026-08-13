package tagbio.umap.fx;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

import javax.imageio.ImageIO;

import junit.framework.TestCase;

public class ImageExportTest extends TestCase {

  private static final int WIDTH = 7;
  private static final int HEIGHT = 5;

  private File mFile;

  @Override
  protected void tearDown() {
    if (mFile != null && !mFile.delete()) {
      mFile.deleteOnExit();
    }
    mFile = null;
  }

  private static int[] gradient() {
    final int[] pixels = new int[WIDTH * HEIGHT];
    for (int y = 0; y < HEIGHT; ++y) {
      for (int x = 0; x < WIDTH; ++x) {
        pixels[y * WIDTH + x] = 0xff000000 | (x * 30 << 16) | (y * 40 << 8) | 0x7f;
      }
    }
    return pixels;
  }

  public void testToBufferedImageKeepsSizeAndPixels() {
    final int[] pixels = gradient();
    final BufferedImage buffer = ImageExport.toBufferedImage(pixels, WIDTH, HEIGHT);
    assertEquals(WIDTH, buffer.getWidth());
    assertEquals(HEIGHT, buffer.getHeight());
    assertEquals(BufferedImage.TYPE_INT_ARGB, buffer.getType());
    for (int y = 0; y < HEIGHT; ++y) {
      for (int x = 0; x < WIDTH; ++x) {
        assertEquals("at " + x + "," + y, pixels[y * WIDTH + x], buffer.getRGB(x, y));
      }
    }
  }

  public void testTooFewPixelsIsRejected() {
    try {
      ImageExport.toBufferedImage(new int[WIDTH * HEIGHT - 1], WIDTH, HEIGHT);
      fail("expected an IllegalArgumentException");
    } catch (final IllegalArgumentException e) {
      assertTrue(e.getMessage(), e.getMessage().contains("pixels"));
    }
  }

  public void testWrittenFileIsAReadablePng() throws IOException {
    mFile = File.createTempFile("image-export-test", ".png");
    final int[] pixels = gradient();
    ImageExport.write(ImageExport.toBufferedImage(pixels, WIDTH, HEIGHT), mFile);

    assertTrue("file is empty", mFile.length() > 0);
    final BufferedImage read = ImageIO.read(mFile);
    assertNotNull("not decodable as an image", read);
    assertEquals(WIDTH, read.getWidth());
    assertEquals(HEIGHT, read.getHeight());
    // PNG is lossless, so every pixel must survive the round trip unchanged.
    for (int y = 0; y < HEIGHT; ++y) {
      for (int x = 0; x < WIDTH; ++x) {
        assertEquals("at " + x + "," + y, pixels[y * WIDTH + x], read.getRGB(x, y));
      }
    }
  }

  public void testWrittenFileStartsWithThePngSignature() throws IOException {
    mFile = File.createTempFile("image-export-test", ".png");
    ImageExport.write(ImageExport.toBufferedImage(gradient(), WIDTH, HEIGHT), mFile);
    final byte[] content = Files.readAllBytes(mFile.toPath());
    final byte[] expected = {(byte) 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    assertTrue("file shorter than the signature", content.length >= expected.length);
    for (int i = 0; i < expected.length; ++i) {
      assertEquals("byte " + i, expected[i], content[i]);
    }
  }
}
