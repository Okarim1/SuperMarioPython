import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.concurrent.ThreadLocalRandom;

import ch.idsia.mario.engine.GlobalOptions;
import ch.idsia.mario.engine.level.Level;
import ch.idsia.mario.engine.level.LevelParser;
import ch.idsia.tools.EvaluationInfo;
import communication.MarioProcess;

public class NewLevel {
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
	
	public static void main(String [] args) throws IOException, InterruptedException
	{
		String command;
		Process p;
		BufferedReader bri;
	    BufferedReader bre;
	    String line;
		File out;
		File file;
		Level level;
		MarioProcess marioProcess = new MarioProcess();
		EvaluationInfo evaluationInfo = new EvaluationInfo();
		int n=200;
		for (int i=35; i<n; i++) {
			command = "python SuperMario_Worlds.py";
			GlobalOptions.Scale = 2;
			GlobalOptions.World = ThreadLocalRandom.current().nextInt(0, 3);;
			p = Runtime.getRuntime().exec(command+" 512_Worlds_Path "+GlobalOptions.World);
		    p.waitFor();
		    bri = new BufferedReader(new InputStreamReader(p.getInputStream()));
		    bre = new BufferedReader(new InputStreamReader(p.getErrorStream()));
	        while ((line = bri.readLine()) != null) {
	            System.out.println(line);
	          }
	          bri.close();
	          while ((line = bre.readLine()) != null) {
	            System.out.println(line);
	          }
	          bre.close();
	          p.waitFor();
	          System.out.println("Done.");
	
		    p.destroy();
			file = new File("testfile.txt");	
			marioProcess.launchMario(new String[0], false); // true means there is a human player
			int j = 0;
			do{
				level = LevelParser.createLevelASCII(file.getPath());
				evaluationInfo = marioProcess.simulateOneLevel(level);
				j++;
			}while(j<10 && evaluationInfo.marioStatus != 1);
			if (evaluationInfo.marioStatus == 1) {
				out = new File("..\\NewLevels\\w"+GlobalOptions.World+"_d"+i+".txt");	
				copyFileUsingStream(file, out);
			}
		}
		System.out.println("Finish");
	}
}
