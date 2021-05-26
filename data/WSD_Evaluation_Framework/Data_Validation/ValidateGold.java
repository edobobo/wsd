import java.io.BufferedReader;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.apache.commons.lang.StringEscapeUtils;

public class ValidateGold
{

	public static void main(String[] args)  throws Exception
	{
	
		if (args.length == 2) {
			checkData(args[0], args[1], false, null);
		
		}else if (args.length == 4 && args[2].equals("wn30")) {
			checkData(args[0], args[1], true, args[3]);
			
		}else 
			exit();
		
		System.out.println("Test finished.");
	}

	private static void exit(){
		System.out.println("ValidateGold data_xml gold_key [wn30 dict_path]\n");
		System.exit(0);
	}
	
	private static void checkData(String pathXML, String pathGoldKey, boolean wn, String dict_path) throws Exception
	{
		
		int numInstances_gold = 0;
		Map<String, Set<String>> map_id2GoldKeys = new HashMap<String, Set<String>>();
		BufferedReader in = Files.newBufferedReader(Paths.get(pathGoldKey), Charset.forName("UTF-8"));
		String line = null;
		System.out.println("Reading gold_key file...");
		while ((line = in.readLine()) != null) 
		{
			numInstances_gold++;
			
			String goldKeys[] = line.split(" ");			
			int prev=-1;
			String lemmaPrev = "";
			Set<String> golds = new HashSet<>();
			for(int i=1;i<goldKeys.length;i++){
				if(wn){		
					int pos = Integer.parseInt(goldKeys[i].substring(goldKeys[i].lastIndexOf("%")+1, goldKeys[i].indexOf(":")));
					if(pos==5)
						pos=3;
					if(prev==-1)
						prev = pos;
					else if(prev!=pos){
						System.err.println("different pos tag in gold key file. line number "+numInstances_gold+"\n"+line);
						System.exit(1);
					}
					
					String lemma = goldKeys[i].substring(0, goldKeys[i].lastIndexOf("%"));
					if(lemmaPrev.isEmpty())
						lemmaPrev = lemma;
					else if (!lemmaPrev.equals(lemma)){
						System.err.println("different lemma in gold key file. line number "+numInstances_gold+"\n"+line);
						System.exit(1);
					}
				}
				golds.add(goldKeys[i]);
			}
			
			String id = goldKeys[0];
			map_id2GoldKeys.put(id, golds);
			
		}
		in.close();
		
		Map<Pair<String,Character>, Set<String>> map_lemmaPos2Keys = null;
		if(wn){
			map_lemmaPos2Keys = new HashMap<>();
			System.out.println("Loading dict file...");
			in = Files.newBufferedReader(Paths.get(dict_path), Charset.forName("UTF-8"));
			while ((line = in.readLine()) != null) 
			{
				Set<String> golds = new HashSet<>();
				String []wn_dict = line.split("\t");
				for(int i=2;i<wn_dict.length;i++){
					golds.add(wn_dict[i]);	
				}
				String lemma = wn_dict[0];
				char pos = wn_dict[1].charAt(0);
				map_lemmaPos2Keys.put(new Pair<String,Character>(lemma, pos), golds);
			}
			in.close();
		}
		
		int numLine=0;
		int numInstances_xml=0;
		System.out.println("Reading data_xml file...");
		in = Files.newBufferedReader(Paths.get(pathXML), Charset.forName("UTF-8"));
		while ((line = in.readLine()) != null) 
		{
			line = line.trim();
			if(line.startsWith("<instance ")){
				String id = line.substring(line.indexOf("id=\"")+4, line.indexOf("\" lemma"));
				String lemma = line.substring(line.indexOf("lemma=\"")+7, line.indexOf("\" pos"));
				lemma = StringEscapeUtils.unescapeXml(lemma);
				String pos = line.substring(line.indexOf("pos=\"")+5, line.indexOf("\">"));
				Set<String> goldKeys = map_id2GoldKeys.get(id);
				if(goldKeys==null){
					System.err.println("instance id not found in gold file! line "+numLine+"\n"+line);
					System.exit(1);
				}
				if(wn){
					String goldKey = goldKeys.iterator().next();
					int posGold = Integer.parseInt(goldKey.substring(goldKey.lastIndexOf("%")+1, goldKey.indexOf(":")));
//					WORDNET POS NUMBER:
//					1    NOUN 
//					2    VERB 
//					3    ADJECTIVE 
//					4    ADVERB 
//					5    ADJECTIVE SATELLITE
					char posWN = 0;
					if(posGold==1){
						posWN = 'n';
						if(!pos.equals("NOUN") && !pos.equals("PROPN")){
							System.err.println("pos in xml and gold file not equals! line "+numLine+"\n"+line);
							System.exit(1);
						}
					}else if(posGold==2){
						posWN = 'v';
						if(!pos.equals("VERB") && !pos.equals("AUX")){
							System.err.println("pos in xml and gold file not equals! line "+numLine+"\n"+line);
							System.exit(1);
						}
					}else if(posGold==3 || posGold==5){
						posWN = 'a';
						if(!pos.equals("ADJ")){
							System.err.println("pos in xml and gold file not equals! line "+numLine+"\n"+line);
							System.exit(1);
						}
					}else if(posGold==4){
						posWN = 'r';
						if(!pos.equals("ADV")){
							System.err.println("pos in xml and gold file not equals! line "+numLine+"\n"+line);
							System.exit(1);
						}
					}
					
					Set<String> possiblesGoldKeys = map_lemmaPos2Keys.get(new Pair<String, Character>(lemma, posWN));
					if(possiblesGoldKeys==null){
						System.err.println("lemma and pos not found in WN30! line "+numLine+"\n"+line);
						System.exit(1);
					}
					for(String s :goldKeys){
						if(!possiblesGoldKeys.contains(s)){
							System.err.println("gold key not retrievable from lemma and pos in WN3.0! line "+numLine+"\n"+line);
							System.exit(1);
						}
					}
					
				}
				numInstances_xml++;
			}
			
			numLine++;
		}
		in.close();
		
		if(numInstances_xml!=numInstances_gold){
			System.err.println("number of instances different between gold and data files");
			System.exit(1);
		}
		
	}
	
	public static class Pair<F, S> 
	{
	    private F first; 
	    private S second;

	    public Pair(F first, S second) {
	        this.first = first;
	        this.second = second;
	    }

	    public void setFirst(F first) {
	        this.first = first;
	    }

	    public void setSecond(S second) {
	        this.second = second;
	    }

	    public F getFirst() {
	        return first;
	    }

	    public S getSecond() {
	        return second;
	    }

		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + ((first == null) ? 0 : first.hashCode());
			result = prime * result + ((second == null) ? 0 : second.hashCode());
			return result;
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			Pair other = (Pair) obj;
			if (first == null) {
				if (other.first != null)
					return false;
			} else if (!first.equals(other.first))
				return false;
			if (second == null) {
				if (other.second != null)
					return false;
			} else if (!second.equals(other.second))
				return false;
			return true;
		}
	    
	    
	}
}
