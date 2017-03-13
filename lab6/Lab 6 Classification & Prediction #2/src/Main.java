import java.util.ArrayList;
import data.*;

public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		//First step load in iris data
		ArrayList<Iris> irisData = DataLoader.LoadAllIrisData();
		System.out.println("Successfully loaded "+irisData.size()+" iris flowers");
		
		//Second step make perceptron or neural network 
	}

}
