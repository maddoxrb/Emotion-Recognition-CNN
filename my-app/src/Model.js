import {
  React,
  useState,
  useRef,
} from 'react'

import { 
  ChakraProvider,
  Center,
  Heading,
  Box,
  VStack,
  Image,
  Card,
  CardBody,
  CardHeader,
  SimpleGrid,
  Divider,
  CardFooter,
  Text,
  AbsoluteCenter,
  Button,
  HStack,
  StackDivider,
  Flex,
  Alert, 
  AlertIcon, 
  AlertDialog, 
  AlertDialogOverlay, 
  AlertDialogContent, 
  AlertDialogHeader, 
  AlertDialogCloseButton, 
  AlertDialogBody, 
  AlertDialogFooter,
  useDisclosure,
  Stack,
  Grid,
  GridItem,
  useBreakpointValue,
} from '@chakra-ui/react'

import UserAccuracyChart from './UserAccuracyChart';
import ModelAccuracyChart from './ModeAccuracyChart';
import InstructionsCard from './InstructionsCard';


// Array that contains the generated images
const generatedImages = [
  { title: 'Generated Image', image: 'https://bit.ly/dan-abramov', model: 'model', actual: 'model'},
  // { title: 'Image 2', image: 'https://bit.ly/dan-abramov', model: 'model', actual: 'actual'},
];

function Model() {
  // const { colorMode } = useColorMode();

  // Button Feature for generating new image
const [isButtonClicked, setIsButtonClicked] = useState(false);
const cancelRef = useRef();
const { isOpen, onOpen, onClose } = useDisclosure();

// Button for determing what the user deems to be the emotion of the user
const [selectedLabel, setSelectedLabel] = useState('');

// Defines the states for altert message, data correctness, and selected label
const [alertMessage, setAlertMessage] = useState('');
const [dataCorrect, setDataCorrect] = useState(0);
const [dataIncorrect, setDataIncorrect] = useState(0);

const data = [
  { name: 'Correct', value: dataCorrect },
  { name: 'Incorrect', value: dataIncorrect },
];

// Get the Grid Template Columns for the middle row
const getGridTemplateColumns = (columnCount) => {
  if (columnCount === 3) {
    return '3fr 4fr 3fr'; // Middle card takes up 2 times the space
  } else {
    return `repeat(${columnCount}, 1fr)`;
  }
};

const handleClick = async () => {
  const runScript = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5000/run-script', {
        method: 'POST',
      });
      const data = await response.json();
      console.log(data);
      return data.result;
    } catch (error) {
      console.error('Error:', error);
      throw error;
    }
  };

  setIsButtonClicked(true);


  if (selectedLabel === '') {
    setAlertMessage('Please select an emotion classification before confirming and generating a new image.');
    onOpen(); // Move onOpen() inside else block
  } else if (selectedLabel === 'angry') {
    const result = await runScript();
    setDataCorrect(dataCorrect + parseInt(result));
    setSelectedLabel('');
  } else {
    setDataIncorrect(dataIncorrect + 1);
    setSelectedLabel('');
    setAlertMessage('');
  }
};

const columnCount = useBreakpointValue({ base: 1, sm: 1, md: 3 });

  return (
    <ChakraProvider>
      <SimpleGrid >
        <Box>
          <Center>
            <Heading pt={5} pb={0} as='h2' size='xl'>
              Model Accuracy Testing
            </Heading>
          </Center>
        </Box>
        <InstructionsCard/>
        <Box display="grid" gridTemplateColumns={getGridTemplateColumns(columnCount)} gap={4} p={10} pt={0}>
          {/* Render the boxes */}
          <Card height="600px">
            <Center>
                <CardHeader>
                  <Heading size='lg'>Model Accuracy</Heading>
                </CardHeader>
              </Center>
            <AbsoluteCenter>
              <ModelAccuracyChart data={data} />
            </AbsoluteCenter>
          </Card>
          <Card largeScreenWidth={columnCount === 3}>
            Title
          </Card>
          <Card>
            Title
          </Card>
        </Box>

      </SimpleGrid>

        <Center pt={10} pb={6}>
        <Button onClick={handleClick} colorScheme='blue'> 
          Confirm and Generate New Image 
        </Button>
        <AlertDialog isOpen={isOpen} leastDestructiveRef={cancelRef} onClose={onClose}>
          <AlertDialogOverlay>
            <AlertDialogContent>
              <AlertDialogHeader fontSize="lg" fontWeight="bold">
                Oops!
              </AlertDialogHeader>
              <AlertDialogCloseButton />
              <AlertDialogBody>
                {alertMessage}
              </AlertDialogBody>
              <AlertDialogFooter>
                <Button ref={cancelRef} onClick={onClose}>
                  Close
                </Button>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialogOverlay>
        </AlertDialog>
      </Center>
    </ChakraProvider>
  
  );
}

export default Model;