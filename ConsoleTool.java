import java.util.*;
import java.io.*;

// this class is used to create a console interface for the user to interact with the program (similar to a command line interface)
// it will have functions to add custom commands that execute their own code and take in their own arguments.
public class ConsoleTool {
    // the scanner object that will be used to read user input
    private Scanner scanner;
    // the hashmap that will store the commands and their respective functions
    private HashMap<String, Command> commands;
    // title of the console
    private String title;
    private InputStream in;
    private PrintStream out;

    // constructor
    public ConsoleTool(InputStream in, PrintStream out, String title) {
        scanner = new Scanner(in);
        commands = new HashMap<String, Command>();
        this.in = in;
        this.out = out;
        this.title = title;
        Clear();
    }

    // this function is used to add a command to the hashmap
    public void addCommand(String commandName, Command command) {
        commands.put(commandName, command);
    }

    // this function is used to start the console interface
    public void start() {
        // the main loop of the console interface
        while (true) {
            // read the user input
            String input = Input();
            // split the input into an array of strings
            String[] inputArray = input.split(" ");
            // get the command name
            String commandName = inputArray[0];
            // get the command arguments
            String[] arguments = Arrays.copyOfRange(inputArray, 1, inputArray.length);
            // check if the command exists in the hashmap
            if (commands.containsKey(commandName)) {
                // execute the command
                commands.get(commandName).execute(arguments);
            } else {
                // if the command does not exist, print an error message
                Output("Error: Command not found");
            }
        }
    }

    // this interface is used to define the structure of a command
    public interface Command {
        // this function is used to execute the command
        public void execute(String... arguments);
    }

    // CALL THIS FUNCTION TO STOP THE CONSOLE INTERFACE
    public void finish() {
        scanner.close();
    }

    public void Output(Object output) {
        out.println(output);
    }

    public void Output(int output) {
        out.println(output);
    }

    public void Output(double output) {
        out.println(output);
    }

    public void Output(int[] arr) {
        //rewrite with stringbuilder
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < arr.length - 1; i++) {
            sb.append(arr[i]);
            sb.append(", ");
        }
        sb.append(arr[arr.length - 1]);
        sb.append("]");
        out.println(sb.toString());
	}

	public void Output(double[] arr) {
		//rewrite with stringbuilder
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < arr.length - 1; i++) {
            sb.append(arr[i]);
            sb.append(", ");
        }
        sb.append(arr[arr.length - 1]);
        sb.append("]");
        out.println(sb.toString());
	}

	public void Output(Object[] arr) {
		//rewrite with stringbuilder
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < arr.length - 1; i++) {
            sb.append(arr[i].toString());
            sb.append(", ");
        }
        sb.append(arr[arr.length - 1]);
        sb.append("]");
        out.println(sb.toString());
	}

    public String Input() {
        System.out.print(">> ");
        return scanner.nextLine();
    }

    public void Clear() {
        out.print("\033[H\033[2J");
        out.flush();
        out.println(title);
    }
}
