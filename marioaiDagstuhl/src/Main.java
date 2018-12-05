import ch.idsia.mario.engine.GlobalOptions;
import ch.idsia.mario.engine.level.Level;
import ch.idsia.mario.engine.level.LevelParser;
import ch.idsia.tools.EvaluationInfo;
import communication.MarioProcess;
import java.io.File;
import java.util.Random;

public class Main {
	public static void main(String [] args)
	{
		EvaluationInfo evaluationInfo = new EvaluationInfo();
		File dir = new File("..\\ComputerLevels");
		
		File[] files = dir.listFiles();
		Random rand = new Random();
		GlobalOptions.Scale = 2;
		MarioProcess marioProcess = new MarioProcess();
		marioProcess.launchMario(new String[0], true); // true means there is a human player
		marioProcess.launchMario(new String[0], true); // true means there is a human player
		while(true) {
			File file = files[rand.nextInt(files.length)];
			System.out.print(file.getPath());
			Level level = LevelParser.createLevelASCII(file.getPath());
			if ( file.getName().charAt(0) == 'w'){
				GlobalOptions.World = Character.getNumericValue(file.getName().charAt(1));
			}else {
				GlobalOptions.World = 0;
			}
			evaluationInfo = marioProcess.simulateOneLevel(level);
			System.out.print(evaluationInfo);
		}
	}
}