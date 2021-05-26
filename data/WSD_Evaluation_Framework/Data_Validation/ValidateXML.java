import java.io.IOException;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;
import javax.xml.transform.Source;
import javax.xml.transform.stream.StreamSource;
import javax.xml.validation.SchemaFactory;
import org.xml.sax.InputSource;
import org.xml.sax.SAXException;
import org.xml.sax.XMLReader;

public class ValidateXML 
{
	public static void main(String[] args)
	{
    	
		// Check command-line arguments.
		if (args.length != 2) {
			exit();
		}
		
		//checking xml
		checkXML(args[0]);
		//checking xml on external xsd
		checkExternalSchema(args[0], args[1]);
		
		System.out.println("Test finished.");
	}
	
	private static void exit() {
		System.out.println("ValidateXML data_xml schema_xsd\n");
		System.exit(0);
	}
	
	private static void checkXML(String pathXML)
	{
		SAXParserFactory factory = SAXParserFactory.newInstance();
		factory.setValidating(false);
		factory.setNamespaceAware(true);
		try{
			SAXParser parser = factory.newSAXParser();
			XMLReader reader = parser.getXMLReader();
			reader.parse(new InputSource(pathXML));		
		} catch (ParserConfigurationException e) {
            e.printStackTrace();
            
            
        } catch (SAXException e) {
            e.printStackTrace();
            
            
        } catch (IOException e) {
            e.printStackTrace();
            
        }
	}
	
	private static void checkExternalSchema(String pathXML, String pathXSD)
	{
        try{
            SAXParserFactory factory = SAXParserFactory.newInstance();
            factory.setValidating(false);
            factory.setNamespaceAware(true);

            SchemaFactory schemaFactory = SchemaFactory.newInstance("http://www.w3.org/2001/XMLSchema");

            factory.setSchema(schemaFactory.newSchema(new Source[] {new StreamSource(pathXSD)}));

            SAXParser parser = factory.newSAXParser();

            XMLReader reader = parser.getXMLReader();
            reader.parse(new InputSource(pathXML));
        } catch (ParserConfigurationException e) {
            e.printStackTrace();
           
        } catch (SAXException e) {
            e.printStackTrace();
           
        } catch (IOException e) {
            e.printStackTrace();
           
        }	
	}
}


