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
  AbsoluteCenter,
  Button,
  AlertDialog, 
  AlertDialogOverlay, 
  AlertDialogContent, 
  AlertDialogHeader, 
  AlertDialogCloseButton, 
  AlertDialogBody, 
  AlertDialogFooter,
  useDisclosure,
} from '@chakra-ui/react'

import UserAccuracyChart from './UserAccuracyChart';
import ModelAccuracyChart from './ModeAccuracyChart';
import InstructionsCard from './InstructionsCard';


// Array that contains the generated images
const generatedImages = [
  { title: 'Generated Image', image: 'https://bit.ly/dan-abramov', model: 'model', actual: 'model'},
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
        <Box>
          <Center>
            <Heading pt={5} pb={0} as='h2' size='xl'>
              Model Accuracy Testing
            </Heading>
          </Center>
        </Box>
        <InstructionsCard/>
      <SimpleGrid columns={[1, 1, 2, 2, 2]} >
        <Box display="grid" gap={4} p={10} pt={0} alignItems='center'>
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
                      <SimpleGrid columns={4} spacing={3} zIndex={2}>
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
      
        </Box>
        <Card height="625px">
            <Center>
              <CardHeader>
                <Heading size='lg'>Your Accuracy</Heading>
              </CardHeader>
            </Center>
            <AbsoluteCenter>
              <UserAccuracyChart data={data} cardWidth={cardWidth}/>
            </AbsoluteCenter>
          </Card>
          
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


      <SimpleGrid columns={[1, 1, 2, 2, 3]}  spacingX='40px' spacingY='20px' p={5}> 
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
      </SimpleGrid>
    </ChakraProvider>
  );
}

export default Model;
