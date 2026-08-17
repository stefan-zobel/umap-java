/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

/**
 * Tests the corresponding class.
 * @author Sean A. Irvine
 */
public class CsrMatrixTest extends AbstractMatrixTest {

  Matrix getMatrixA() {
    return new DefaultMatrix(new float[][] {{0, 1}, {0.5F, 2}, {1, 0}, {0, 3}}).toCsr();
  }

  // Don't test functionality not yet supported in Csr
  @Override
  public void testAdd() {
  }

  @Override
  public void testSubtract() {
  }

  @Override
  public void testEquals() {
  }

  @Override
  public void testMultiply() {
  }

  @Override
  public void testTranspose() {
  }

  /**
   * <code>intersect</code> writes its whole result into the supplied matrix. Before the fix
   * it wrote into the copy handed back by <code>CooMatrix.data()</code>, so
   * <code>Umap.generalSimplicialSetIntersection</code> silently returned the plain sum of
   * the two simplicial sets instead of their intersection.
   */
  public void testIntersectWritesIntoResult() {
    final Matrix left = new DefaultMatrix(new float[][] {{1, 2}, {3, 4}});
    final Matrix right = new DefaultMatrix(new float[][] {{5, 6}, {7, 8}});
    final CooMatrix result = left.toCoo().add(right.toCoo()).toCoo();

    // The sum, i.e. what the caller sees if intersect writes nowhere.
    assertEquals(6.0F, result.get(0, 0));
    assertEquals(8.0F, result.get(0, 1));
    assertEquals(10.0F, result.get(1, 0));
    assertEquals(12.0F, result.get(1, 1));

    // With mixWeight 0.5 the formula collapses to leftVal * rightVal.
    left.toCsr().intersect(right.toCsr(), result, 0.5F);

    assertEquals(5.0F, result.get(0, 0));
    assertEquals(12.0F, result.get(0, 1));
    assertEquals(21.0F, result.get(1, 0));
    assertEquals(32.0F, result.get(1, 1));
  }
}
