package tagbio.umap.fx;

import javafx.application.Application;

/**
 * Entry point for the UMAP viewer.
 *
 * <p>This class deliberately does not extend {@link Application}. When JavaFX is a plain
 * classpath dependency, as it is when the project is run from the IDE, a main method
 * declared in an Application subclass aborts with "JavaFX runtime components are missing".
 * Going through a separate launcher works both there and when JavaFX is put on the module
 * path, as the exec plugin does.
 */
public final class Launcher {

  private Launcher() { }

  /**
   * Start the viewer.
   * @param args the first non-empty element is the path of the data file to project
   */
  public static void main(final String[] args) {
    Application.launch(UmapViewer.class, args);
  }
}
