package tagbio.umap.fx;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import javafx.scene.image.Image;
import javafx.scene.image.PixelFormat;
import javafx.scene.image.PixelReader;

/**
 * Writes a rendered node to a PNG file.
 *
 * <p>The pixels are copied through a plain integer array rather than through
 * <code>SwingFXUtils</code>, which lives in the separate <code>javafx-swing</code> artifact
 * that this project does not depend on. Only the JDK's own image writer is needed.
 */
final class ImageExport {

  private ImageExport() { }

  /**
   * Write an image to a PNG file.
   * @param image the rendered image
   * @param file destination, overwritten if it exists
   * @throws IOException if the file cannot be written or no PNG writer is registered
   */
  static void writePng(final Image image, final File file) throws IOException {
    final int width = (int) Math.round(image.getWidth());
    final int height = (int) Math.round(image.getHeight());
    if (width <= 0 || height <= 0) {
      throw new IOException("Nothing to export");
    }
    final PixelReader reader = image.getPixelReader();
    if (reader == null) {
      throw new IOException("Image cannot be read back");
    }
    final int[] pixels = new int[width * height];
    reader.getPixels(0, 0, width, height, PixelFormat.getIntArgbInstance(), pixels, 0, width);
    write(toBufferedImage(pixels, width, height), file);
  }

  /**
   * Wrap packed ARGB pixels in a buffered image. Separate from the JavaFX side so that the
   * conversion and the file writing can be exercised without starting a toolkit.
   * @param pixels packed ARGB values, row major
   * @param width image width
   * @param height image height
   * @return the buffered image
   */
  static BufferedImage toBufferedImage(final int[] pixels, final int width, final int height) {
    if (pixels.length < width * height) {
      throw new IllegalArgumentException("Expected " + width * height + " pixels, got " + pixels.length);
    }
    final BufferedImage buffer = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
    buffer.setRGB(0, 0, width, height, pixels, 0, width);
    return buffer;
  }

  /**
   * Write a buffered image out as a PNG.
   * @param buffer the image
   * @param file destination, overwritten if it exists
   * @throws IOException if the file cannot be written or no PNG writer is registered
   */
  static void write(final BufferedImage buffer, final File file) throws IOException {
    if (!ImageIO.write(buffer, "png", file)) {
      throw new IOException("No PNG writer available");
    }
  }
}
