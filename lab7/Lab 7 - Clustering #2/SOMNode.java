/**
 *
 * @author  alanter
 */
public class SOMNode {
	private SOMVector weights;
	private int xp, yp;
	
	/** Creates a new instance of SOMNode */
	public SOMNode(int numWeights) {
		weights = new SOMVector();
		for (int x=0; x<numWeights; x++) {
			weights.addElement(new Double(Math.random()));
		}
	}
	
	public void setX(int xpos) {
		xp = xpos;
	}
	
	public void setY(int ypos) {
		yp = ypos;
	}
	
	public int getX() {
		return xp;
	}
	
	public int getY() {
		return yp;
	}
	
	/** Computes the distance to another node.  Used for
	 *  neighborhood determination.  Returns the SQUARE of the distance
	 */
	public double distanceTo(SOMNode n2) {
		int xleg, yleg;
		xleg = getX() - n2.getX();
		xleg *= xleg;
		yleg = getY() - n2.getY();
		yleg *= yleg;
		return xleg + yleg;
	}
	
	public void setWeight(int w, double value) {
		if (w >= weights.size())
			return;
		weights.setElementAt(new Double(value), w);
	}
	
	public double getWeight(int w) {
		if (w >= weights.size())
			return 0;
		
		return ((Double)weights.elementAt(w)).doubleValue();
	}
	
	public SOMVector getVector() {
		return weights;
	}
	
	public void adjustWeights(SOMVector input, double learningRate,
							  double distanceInfluence)
	{
		
	}
}