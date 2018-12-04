import ch.idsia.mario.engine.level.Level;
import ch.idsia.mario.engine.level.LevelParser;
import communication.MarioProcess;

public class Main {
	public static void main(String [] args)
	{
		Level level = LevelParser.createLevelASCII("../ComputerLevels/14.txt");
		MarioProcess marioProcess = new MarioProcess();
		marioProcess.launchMario(new String[0], true); // true means there is a human player       
		marioProcess.simulateOneLevel(level);
	}
}
