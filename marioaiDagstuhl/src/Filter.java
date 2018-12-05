import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import ch.idsia.mario.engine.GlobalOptions;
import ch.idsia.mario.engine.level.Level;
import ch.idsia.mario.engine.level.LevelParser;
import ch.idsia.tools.EvaluationInfo;
import communication.MarioProcess;

public class Filter {
	private static void copyFileUsingStream(File source, File dest) throws IOException {
	    InputStream is = null;
	    OutputStream os = null;
	    try {
	        is = new FileInputStream(source);
	        os = new FileOutputStream(dest);
	        byte[] buffer = new byte[1024];
	        int length;
	        while ((length = is.read(buffer)) > 0) {
	            os.write(buffer, 0, length);
	        }
	    } finally {
	        is.close();
	        os.close();
	    }
	}
	
	public static void main(String [] args) throws IOException
	{
		File dir = new File("..\\ComputerLevels");	
		File out;
		int i;
		Level level;
		File[] files = dir.listFiles();
		GlobalOptions.Scale = 2;
		GlobalOptions.World = 0;
		MarioProcess marioProcess = new MarioProcess();
		EvaluationInfo evaluationInfo = new EvaluationInfo();
		marioProcess.launchMario(new String[0], false); // true means there is a human player
		for(int j=0; j<files.length; j++) {
			i = 0;
			System.out.println(files[j].getName());
			do{
				level = LevelParser.createLevelASCII(files[j].getPath());
				evaluationInfo = marioProcess.simulateOneLevel(level);
				i++;
			}while(i<5 && evaluationInfo.marioStatus != 1);
			if (evaluationInfo.marioStatus == 1) {
				out = new File("..\\FilterLevels\\"+files[j].getName());	
				copyFileUsingStream(files[j], out);
			}
		}
		System.out.println("Finish");
	}
}
