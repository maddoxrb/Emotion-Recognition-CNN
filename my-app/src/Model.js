import {
  React,
  useState,
  useRef,
  useEffect
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

import { FaCircle } from 'react-icons/fa';

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
    return '6fr 3fr 3fr'; // Middle card takes up 2 times the space
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

const columnCount = useBreakpointValue({ base: 1, sm: 1, md: 1, lg: 3 });

// State to hold the width of the card
const [cardWidth, setCardWidth] = useState(0);

// Ref to the card element
const cardRef = useRef(null);

// Update the card width when the component mounts or the card's width changes
useEffect(() => {
  const updateCardWidth = () => {
    const width = cardRef.current.clientWidth;
    setCardWidth(width);
  };

  updateCardWidth();
  window.addEventListener('resize', updateCardWidth);

  return () => {
    window.removeEventListener('resize', updateCardWidth);
  };
}, []);

  return (
    <ChakraProvider>
      <SimpleGrid>
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
          {generatedImages.map((images, index) => (
            <Card key={index} height="625px" width="100%" ref={cardRef}>
              <Center>
                <CardHeader>
                  <Heading size='lg'>{images.title}</Heading>
                </CardHeader>
              </Center>
              <Center>
                <CardBody mt={-5}>
                  <Center>
                    <Image
                      src={images.image}
                      borderRadius='lg'
                    />
                  </Center>
                </CardBody>
              </Center>
              <Box position='relative' padding='6'>
                <Divider />
              </Box>
              <Center>
                <CardFooter>
                  <VStack spacing="10px">
                    <Center>
                      <SimpleGrid columns={4} spacing={6}>
                        <Button
                          onClick={() => setSelectedLabel('angry')}
                          colorScheme={selectedLabel === 'angry' ? 'blue' : 'gray'}
                          flex="1"
                        >
                          Angry
                        </Button>
                        <Button
                          onClick={() => setSelectedLabel('disgust')}
                          colorScheme={selectedLabel === 'disgust' ? 'blue' : 'gray'}
                          flex="1"
                        >
                          Disgust
                        </Button>
                        <Button
                          onClick={() => setSelectedLabel('fear')}
                          colorScheme={selectedLabel === 'fear' ? 'blue' : 'gray'}
                          flex="1"
                        >
                          Fear
                        </Button>
                        <Button
                          onClick={() => setSelectedLabel('happy')}
                          colorScheme={selectedLabel === 'happy' ? 'blue' : 'gray'}
                          flex="1"
                        >
                          Happy
                        </Button>
                        <Button
                          onClick={() => setSelectedLabel('natural')}
                          colorScheme={selectedLabel === 'natural' ? 'blue' : 'gray'}
                          flex="1"
                        >
                          Natural
                        </Button>
                        <Button
                          onClick={() => setSelectedLabel('sad')}
                          colorScheme={selectedLabel === 'sad' ? 'blue' : 'gray'}
                          flex="1"
                        >
                          Sad
                        </Button>
                        <Button
                          onClick={() => setSelectedLabel('surprise')}
                          colorScheme={selectedLabel === 'surprise' ? 'blue' : 'gray'}
                          flex="1"
                        >
                          Surprise
                        </Button>
                      </SimpleGrid>
                    </Center>
                  </VStack>
                </CardFooter>
              </Center>
      
            </Card>
          ))}
          <Card height="625px">
            <Center>
              <CardHeader>
                <Heading size='lg'>Your Accuracy</Heading>
              </CardHeader>
            </Center>
            <AbsoluteCenter>
              <UserAccuracyChart data={data} cardWidth={cardWidth}/>
            </AbsoluteCenter>
            <CardFooter position="absolute" bottom="0" width="100%" p={5} justifyContent="center">
              <Flex justifyContent="center" alignItems="center" flexDirection="column">
                <Flex alignItems="center" mb={2} justifyContent="center">
                  <FaCircle size={10} color="#107C10" />
                  <Text fontWeight="bold" color="#107C10" ml={1} fontSize={20}>
                    Correct
                  </Text>
                </Flex>
                <Flex alignItems="center" justifyContent="center">
                  <FaCircle size={10} color="#D80000" />
                  <Text fontWeight="bold" color="#D80000" ml={1} fontSize={20}>
                    Incorrect
                  </Text>
                </Flex>
              </Flex>
            </CardFooter>
          </Card>
          <Card height="625px">
            <Center>
              <CardHeader>
                <Heading size='lg'>Model Accuracy</Heading>
              </CardHeader>
            </Center>
            <AbsoluteCenter>
              <ModelAccuracyChart data={data} />
            </AbsoluteCenter>
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
