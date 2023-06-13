import {
  React,
  useState,
  useRef,
} from 'react';

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
} from '@chakra-ui/react';

import UserAccuracyChart from './UserAccuracyChart';
import ModelAccuracyChart from './ModeAccuracyChart';
import InstructionsCard from './InstructionsCard';

// Array that contains the generated images
const generatedImages = [
  { title: 'Generated Image', image: 'https://bit.ly/dan-abramov', model: 'model', actual: 'model' },
  // { title: 'Image 2', image: 'https://bit.ly/dan-abramov', model: 'model', actual: 'actual'},
];

function Model() {
  // Button Feature for generating new image
  const [isButtonClicked, setIsButtonClicked] = useState(false);
  const cancelRef = useRef();
  const { isOpen, onOpen, onClose } = useDisclosure();

  // Button for determining what the user deems to be the emotion of the user
  const [selectedLabel, setSelectedLabel] = useState('');

  // Defines the states for alert message, data correctness, and selected label
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
      onOpen();
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

  return (
    <ChakraProvider>
      <Center>
        <Heading pt={5} pb={0} as="h2" size="xl">
          Model Accuracy Testing
        </Heading>
      </Center>
      <InstructionsCard />
      <Flex height="80vh" direction={['column', 'column', 'row']} justifyContent="space-between">
        <Flex width={['100%', '100%', '30%']} justifyContent="center" alignItems="center">
          <Card height="100%" width="90%">
            <Center>
              <CardHeader>
                <Heading size="lg">Model Accuracy</Heading>
              </CardHeader>
            </Center>
            <AbsoluteCenter>
              <ModelAccuracyChart data={data} />
            </AbsoluteCenter>
          </Card>
        </Flex>
        <Flex width={['100%', '100%', '40%']} justifyContent="center" alignItems="center">
          <SimpleGrid m={10} spacing={10} templateColumns="repeat(auto-fill, minmax(400px, 2fr))">
            {generatedImages.map((images, index) => (
              <Card key={index} height="100%" width="100%">
                <Center>
                  <CardHeader>
                    <Heading size="lg">{images.title}</Heading>
                  </CardHeader>
                </Center>
                <Center>
                  <CardBody mt={-5}>
                    <Center>
                      <Image src={images.image} borderRadius="lg" />
                    </Center>
                  </CardBody>
                </Center>
                <Box position="relative" padding="6">
                  <Divider />
                </Box>
                <Center>
                  <CardFooter>
                    <VStack spacing="10px">
                      <Center>
                        <SimpleGrid columns={[2, 2, 4]} spacing={6}>
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
          </SimpleGrid>
        </Flex>
        <Flex width={['100%', '100%', '30%']} justifyContent="center" alignItems="center">
          <Card height="100%" width="90%">
            <Center>
              <CardHeader>
                <Heading size="lg">Your Accuracy</Heading>
              </CardHeader>
            </Center>
            <AbsoluteCenter>
              <UserAccuracyChart data={data} />
            </AbsoluteCenter>
          </Card>
        </Flex>
      </Flex>
      <Center pt={10} pb={6}>
        <Button onClick={handleClick} colorScheme="blue">
          Confirm and Generate New Image
        </Button>
        <AlertDialog
          motionPreset="slideInBottom"
          leastDestructiveRef={cancelRef}
          onClose={onClose}
          isOpen={isOpen}
          isCentered
        >
          <AlertDialogOverlay />

          <AlertDialogContent>
            <AlertDialogHeader>Alert</AlertDialogHeader>
            <AlertDialogCloseButton />
            <AlertDialogBody>{alertMessage}</AlertDialogBody>
            <AlertDialogFooter>
              <Button ref={cancelRef} onClick={onClose}>
                Close
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </Center>
    </ChakraProvider>
  );
}

export default Model;
