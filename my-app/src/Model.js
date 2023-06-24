import React, { useState, useRef, useEffect } from 'react';
import {
  ChakraProvider,
  Center,
  Heading,
  Box,
  VStack,
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
  useToast, // Import useToast hook
} from '@chakra-ui/react';

import UserAccuracyChart from './UserAccuracyChart';
import Model1AccuracyChart from './Model1AccuracyChart';
import Model2AccuracyChart from './Model2AccuracyChart';
import Model3AccuracyChart from './Model3AccuracyChart';
import InstructionsCard from './InstructionsCard';

// Array that contains the generated images
const generatedImages = [
  { title: 'Generated Image', image: 'https://bit.ly/dan-abramov', model: 'model', actual: 'model' },
];

function Model() {
  // Set the default image URL here
  const [imageUrl, setImageUrl] = useState(require('./FLOWERS/bellflowers/bellflowers1.jpg'));

  // Button Feature for generating new image
  const [isButtonClicked, setIsButtonClicked] = useState(false);
  const cancelRef = useRef();
  const { isOpen, onOpen, onClose } = useDisclosure();

  // Button for determining what the user deems to be the emotion of the user
  const [selectedLabel, setSelectedLabel] = useState('');

  // Defines the states for alert message, data correctness, and selected label
  const [alertMessage, setAlertMessage] = useState('');
  const [usrDataCorrect, setDataCorrect] = useState(0);
  const [usrDataIncorrect, setDataIncorrect] = useState(0);
  const [model1DataCorrect, setmodel1DataCorrect] = useState(0);
  const [model1DataIncorrect, setmodel1DataIncorrect] = useState(0);
  const [model2DataCorrect, setmodel2DataCorrect] = useState(0);
  const [model2DataIncorrect, setmodel2DataIncorrect] = useState(0);
  const [model3DataCorrect, setmodel3DataCorrect] = useState(0);
  const [model3DataIncorrect, setmodel3DataIncorrect] = useState(0);

  // Define our previous image
  const [prevAnswer, setPrevAnswer] = useState('bellflower')

  // Store the PieChart Data
  const userData = [
    { name: 'Correct', value: usrDataCorrect },
    { name: 'Incorrect', value: usrDataIncorrect },
  ];
  const model1Data = [
    { name: 'Correct', value: model1DataCorrect },
    { name: 'Incorrect', value: model1DataIncorrect },
  ];
  const model2Data = [
    { name: 'Correct', value: model2DataCorrect },
    { name: 'Incorrect', value: model2DataIncorrect },
  ];
  const model3Data = [
    { name: 'Correct', value: model3DataCorrect },
    { name: 'Incorrect', value: model3DataIncorrect },
  ];

  // Create a toast notification
  const toast = useToast();

  const handleClick = async () => {
    const runScript = async () => {
      try {
        const response = await fetch('http://34.139.34.233:8000/run_script', {
          method: 'POST',
          mode: 'cors',
        });
        const data = await response.json();
        console.log(data);
        return data;
      } catch (error) {
        console.error('Error1:', error);
        throw error;
      }
    };

    console.log(isButtonClicked)
    setIsButtonClicked(true);

    if (selectedLabel === '') {
      setAlertMessage('Please select a flower classification before confirming and generating a new image.');
      onOpen();
    } else {
      try {
        const response = await runScript();
        console.log(response);

        // Access the values from the JSON response
        const {
          answer,
          model1correct,
          model1incorrect,
          model2correct,
          model2incorrect,
          model3correct,
          model3incorrect,
          filename,
        } = response;

        console.log(model1correct);

        // Handle the parsed JSON data here
        if (selectedLabel === prevAnswer) {
          setDataCorrect(usrDataCorrect + 1);
        } else {
          setDataIncorrect(usrDataIncorrect + 1);

          // Show toast notification with the actual answer
          toast({
            title: 'Incorrect Answer',
            description: `The actual answer was ${prevAnswer}.`,
            status: 'error',
            duration: 5000,
            position: 'top-left',
            isClosable: true,
          });
        }

        // Update the image URL with the new image
        const newImageUrl = require(`${filename}`);
        setImageUrl(newImageUrl);

        // Update new answer
        setPrevAnswer(answer)

        // Update the model accuracies
        setmodel1DataCorrect(model1DataCorrect + model1correct);
        setmodel1DataIncorrect(model1DataIncorrect + model1incorrect);
        setmodel2DataCorrect(model2DataCorrect + model2correct);
        setmodel2DataIncorrect(model2DataIncorrect + model2incorrect);
        setmodel3DataCorrect(model3DataCorrect + model3correct);
        setmodel3DataIncorrect(model3DataIncorrect + model3incorrect);

        setSelectedLabel('');
        setAlertMessage('');
      } catch (error) {
        console.error('Error2:', error);
      }
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
          <Heading pt={5} pb={0} as="h2" size="xl">
            Model Accuracy Testing
          </Heading>
        </Center>
      </Box>
      <InstructionsCard />
      <SimpleGrid columns={[1, 1, 2, 2, 2]}>
        <Box display="grid" gap={4} p={5} pt={0} alignItems="center">
          {/* Render the boxes */}
          {generatedImages.map((images, index) => (
            <Card key={index} height="625px" width="100%" ref={cardRef}>
              <Center>
                <CardHeader>
                  <Heading size="lg">{images.title}</Heading>
                </CardHeader>
              </Center>
              <Center>
                <CardBody mt={-5}>
                  <Center>
                    <Box
                      width="350px" // Set the desired container width
                      height="325px" // Set the desired container height
                      overflow="hidden"
                      borderRadius="lg" // Use "lg" for rounded corners
                    >
                      <img
                        src={imageUrl}
                        alt=""
                        style={{ width: "100%", height: "100%", objectFit: "cover" }}
                      />
                    </Box>
                  </Center>
                </CardBody>
              </Center>
              <Box position="relative" padding="5">
                <Divider />
              </Box>
              <Center>
                <CardFooter>
                  <VStack spacing="10px">
                    <Center>
                      <SimpleGrid columns={4} spacing={2} zIndex={2}>
                        <Button
                          onClick={() => setSelectedLabel("bellflower")}
                          colorScheme={selectedLabel === "bellflower" ? "blue" : "gray"}
                          flex="1"
                        >
                          Bellflower
                        </Button>
                        <Button
                          onClick={() => setSelectedLabel("daisy")}
                          colorScheme={selectedLabel === "daisy" ? "blue" : "gray"}
                          flex="1"
                        >
                          Daisy
                        </Button>
                        <Button
                          onClick={() => setSelectedLabel("dandelion")}
                          colorScheme={selectedLabel === "dandelion" ? "blue" : "gray"}
                          flex="1"
                        >
                          Dandelion
                        </Button>
                        <Button
                          onClick={() => setSelectedLabel("lotus")}
                          colorScheme={selectedLabel === "lotus" ? "blue" : "gray"}
                          flex="1"
                        >
                          Lotus
                        </Button>
                        <Button
                          onClick={() => setSelectedLabel("rose")}
                          colorScheme={selectedLabel === "rose" ? "blue" : "gray"}
                          flex="1"
                        >
                          Rose
                        </Button>
                        <Button
                          onClick={() => setSelectedLabel("sunflower")}
                          colorScheme={selectedLabel === "sunflower" ? "blue" : "gray"}
                          flex="1"
                        >
                          Sunflower
                        </Button>
                        <Button
                          onClick={() => setSelectedLabel("tulips")}
                          colorScheme={selectedLabel === "tulips" ? "blue" : "gray"}
                          flex="1"
                        >
                          Tulips
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
              <Heading size="lg">Your Accuracy</Heading>
            </CardHeader>
          </Center>
          <AbsoluteCenter>
            <UserAccuracyChart data={userData} cardWidth={cardWidth} />
          </AbsoluteCenter>
        </Card>
      </SimpleGrid>

      <Center pt={10} pb={6}>
        <Button onClick={handleClick} colorScheme="blue">
          Confirm and Generate New Image
        </Button>
        <AlertDialog isOpen={isOpen} leastDestructiveRef={cancelRef} onClose={onClose}>
          <AlertDialogOverlay>
            <AlertDialogContent>
              <AlertDialogHeader fontSize="lg" fontWeight="bold">
                Oops!
              </AlertDialogHeader>
              <AlertDialogCloseButton />
              <AlertDialogBody>{alertMessage}</AlertDialogBody>
              <AlertDialogFooter>
                <Button ref={cancelRef} onClick={onClose}>
                  Close
                </Button>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialogOverlay>
        </AlertDialog>
      </Center>

      <SimpleGrid columns={[1, 1, 2, 2, 3]} spacingX="40px" spacingY="20px" p={5}>
        <Card height="625px">
        <Center>
          <CardHeader>
            <Heading size="lg" align="center">
              Custom Convolutional Model Accuracy
            </Heading> 
          </CardHeader>
        </Center>
          <AbsoluteCenter>
            <Model2AccuracyChart data={model2Data} cardWidth={cardWidth} />
          </AbsoluteCenter>
        </Card>
        <Card height="625px">
          <Center>
            <CardHeader>
              <Heading size="lg" align="center">Ensemble Model Accuracy</Heading>
            </CardHeader>
          </Center>
          <AbsoluteCenter>
            <Model3AccuracyChart data={model3Data} cardWidth={cardWidth} />
          </AbsoluteCenter>
        </Card>
        <Card height="625px">
          <Center>
            <CardHeader>
              <Heading size="lg" align="center">Transfer-Learning Model Accuracy</Heading>
            </CardHeader>
          </Center>
          <AbsoluteCenter>
            <Model1AccuracyChart data={model1Data} cardWidth={cardWidth} />
          </AbsoluteCenter>
        </Card>
      </SimpleGrid>
    </ChakraProvider>
  );
}

export default Model;
