import { ChakraProvider, Box, Flex, Text, useColorMode, VStack, Center, Heading, Link, UnorderedList, ListItem, Card, CardHeader, CardBody, HStack, SimpleGrid} from "@chakra-ui/react";

function Info() {
  const { colorMode } = useColorMode();

  return (
    <ChakraProvider>
      <Box>
        <Center>
          <Heading as="h1" size="lg" pt={5}>
            About the Models
          </Heading>
        </Center>
      </Box>
      <Box p={4}>
        <Flex flexWrap="wrap" justifyContent="center">
          {/* Custom Convolutional Model */}
          <VStack m={4} p={4} boxShadow="md" borderRadius="md" bg={colorMode === "dark" ? "gray.800" : "gray.100"}>
            <Text fontSize="xl" fontWeight="bold">Custom Convolutional Model</Text>
            <Text>
              Our first model is a custom-configured convolutional neural network. This means that we designed the model architecture ourselves, without utilizing any pre-existing data for initial weight assignment. Through experimentation with varying numbers of convolutional and fully connected layers, we achieved the highest level of testing and training accuracy with a configuration consisting of two convolutional layers and four fully connected layers. The final model produced a training accuracy of 91.23%, and a testing accuracy of 75.94%.
            </Text>
            <Text fontSize="md" fontWeight="bold" fontStyle="italic">Benefits of Using a Custom Model:</Text>
            <Text>
              <UnorderedList>
                <ListItem>Specific Configuration: A custom model can be tailored precisely to suit the specifications of the dataset at hand.</ListItem>
                <ListItem>Enhanced Domain Learning: Compared to pre-trained generic models, a custom model excels in capturing domain-specific features.</ListItem>
                <ListItem>Computational Efficiency: Custom models can often exhibit superior computational efficiency when compared to other types of models.</ListItem>
              </UnorderedList>
            </Text>
            <Text fontSize="md" fontWeight="bold" fontStyle="italic">Drawbacks of Using a Custom Model:</Text>
            <Text>
              <UnorderedList>
                <ListItem>
                  Data Requirements: Custom models typically demand larger amounts of data since they lack pre-training on existing samples.
                </ListItem>
                <ListItem>
                Overfitting Risks: Smaller datasets can pose a challenge as custom models may become overly fixated on certain features, leading to overfitting issues.
                </ListItem>
                <ListItem>
                Performance Limitations: When compared to benchmark or state-of-the-art models, custom models may display limited performance capabilities.
                </ListItem>
              </UnorderedList>
            </Text>
          </VStack>

          {/* Ensemble Model */}
          <VStack m={4} p={4} boxShadow="md" borderRadius="md" bg={colorMode === "dark" ? "gray.800" : "gray.100"}>
            <Text fontSize="xl" fontWeight="bold">Ensemble Model</Text>
            <Text>
              Our second model employs the technique of 'ensemble learning' to enhance the accuracy and versatility of the initial custom model. This prediction model leverages the architecture of the previous custom model while employing three distinct instances with slightly varied hyperparameters and learning rates. Once all three models complete their training, their weights are combined, and the optimal classification is selected. While we utilized only three models due to limitations in computing power and training time, it is possible to incorporate more. The final model produced a training accuracy of 92.13% and a testing accuracy of 79.33%.
            </Text>
            <Text fontSize="md" fontWeight="bold" fontStyle="italic">Benefits of Using an Ensemble Model:</Text>
            <Text>
              <UnorderedList>
                <ListItem>Enhanced Accuracy and Overfitting Prevention: Proper implementation of ensemble learning can improve the overall accuracy of the model and mitigate overfitting issues.</ListItem>
                <ListItem>Improved Generality: By combining the weights of multiple models, the ensemble model achieves heightened generality in its predictions.</ListItem>
                <ListItem>Robustness through Model Combination: Ensemble learning allows for the integration of models with different types and architectures, resulting in a more resilient and robust model.</ListItem>
              </UnorderedList>
            </Text>
            <Text fontSize="md" fontWeight="bold" fontStyle="italic">Drawbacks of Using an Ensemble Model:</Text>
            <Text>
              <UnorderedList>
                <ListItem>Longer Training Times: Training multiple models individually leads to extended training durations.</ListItem>
                <ListItem>Computational Intensity and Memory Usage: Ensemble models can be computationally demanding and consume significant memory resources.</ListItem>
                <ListItem>Overfitting Risks: Improper implementation of ensemble learning can lead to overfitting problems, necessitating careful attention during model development.</ListItem>
              </UnorderedList>
            </Text>
          </VStack>

          {/* Transfer-Learning Model */}
          <VStack m={4} p={4} boxShadow="md" borderRadius="md" bg={colorMode === "dark" ? "gray.800" : "gray.100"}>
            <Text fontSize="xl" fontWeight="bold">Transfer-Learning Model</Text>
            <Text>
              Our final model utilizes the technique of transfer learning, which involves taking a pretrained model and further training it on a new dataset to capture domain-specific features. In our case, we opted for ResNet-18, a deep learning model pretrained on a vast dataset comprising over a million images. By leveraging the knowledge embedded in these pretrained weights, we achieved a higher starting accuracy and obtained outstanding results. The final model procured a training accuracy of 95.34% and a testing accuracy of 91.07%.
            </Text>
            <Text fontSize="md" fontWeight="bold" fontStyle="italic">Benefits of Using Transfer-Learning:</Text>
            <Text>
              <UnorderedList>
                <ListItem>Improved Performance with Limited Data: Transfer learning excels in scenarios where data availability is limited, as it leverages the pretrained model's knowledge to boost performance.</ListItem>
                <ListItem>Reduced Training Time and Computational Intensity: Utilizing a pretrained model significantly reduces the training time and computational resources required compared to training from scratch.</ListItem>
                <ListItem>Deeper and More Accurate Feature Maps: Transfer learning often yields deeper and more accurate feature maps due to the pretrained model's extensive learning experience.</ListItem>
              </UnorderedList>
            </Text>
            <Text fontSize="md" fontWeight="bold" fontStyle="italic">Drawbacks of Using Transfer-Learning:</Text>
            <Text>
              <UnorderedList>
                <ListItem>Limited Flexibility and Control over Model Architecture: Employing a pretrained model restricts the flexibility and control you have over the model's architecture, as it is already preconfigured.</ListItem>
                <ListItem>Potential Transfer Bias: Transfer learning may introduce transfer bias, where biases present in the pretraining domain carry over to the new training domain, leading to skewed classifications.</ListItem>
                <ListItem>Mismatch between Pretraining and New Training Domains: A discrepancy between the pretraining domain and the new training domain can limit the accuracy of the classification model, as the pretrained model's knowledge may not fully align with the new dataset.</ListItem>
              </UnorderedList>
            </Text>
          </VStack>
        </Flex>
      </Box>
      <Box>
        <Center>
          <Heading  as="h1" size="lg">
            About the Dataset
          </Heading>
        </Center>
      </Box>
      <Box p={4}>
        <Flex flexWrap="wrap" justifyContent="center">
          {/* Custom Convolutional Model */}
          <VStack m={4} p={4} boxShadow="md" borderRadius="md" bg={colorMode === "dark" ? "gray.800" : "gray.100"}>
            <Text>
            All three models were trained on a dataset of 11,200 images of flowers categorized into 7 different categories: Bellflower, Daisy, Rose, Tulip, Sunflower, Lotus, and Dandelion. This dataset was obtained for training under a CC0 Public Domain Licence. Credit for the original dataset: {' '}
            <Link color='blue.500' href='https://www.kaggle.com/datasets/nadyana/flowers'>
            https://www.kaggle.com/datasets/nadyana/flowers
            </Link>
            </Text>
          </VStack>
        </Flex>
      </Box>
      <Box>
        <Center>
          <Heading as="h1" size="lg">
            Addendum
          </Heading>
        </Center>
      </Box>
      <Box p={4}>
        <Flex flexWrap="wrap" justifyContent="center">
          {/* Custom Convolutional Model */}
          <VStack m={4} p={4} boxShadow="md" borderRadius="md" bg={colorMode === "dark" ? "gray.800" : "gray.100"}>
            <Text>
            It is important to note that our model development and training were limited by the computing power at our disposal. Throughout the entire process, including training and testing, we did not have the benefit of GPU support. As a result, training times were significantly prolonged compared to what could have been achieved with CUDA-enabled acceleration. Considering this limitation, it is worth emphasizing that further advancements and refinements in these models are plausible with enhanced computing power. Given the availability of improved resources, the potential for optimizing and enhancing the performance of these models is substantial.
            </Text>
          </VStack>
        </Flex>
      </Box>
      <Box>
        <Center>
          <Heading as="h1" size="lg">
            About Us
          </Heading>
        </Center>
      </Box>
      <Box p={4}>
        <Flex flexWrap="wrap" justifyContent="center">
          {/* Custom Convolutional Model */}
          <VStack m={4} p={4} boxShadow="md" borderRadius="md" bg={colorMode === "dark" ? "gray.800" : "gray.100"}>
            <Text>
            Aditya Shrey and Maddox Barron, both currently students at Vanderbilt University, studying Computer Science worked on the Flower Classifier.

            For any questions regarding additional details about our models or the project, feel free to reach out to us using the provided contact information below:
            </Text>

            Aditya Shrey
            Vanderbilt University
            Email: aditya.shrey@vanderbilt.edu

            <SimpleGrid p={5} columns={[1, 1, 2, 2, 2]} spacing={3}>
            <Card maxWidth="400px">
              <CardHeader fontWeight="bold" fontSize="xl">Maddox Barron</CardHeader>
              <CardBody>
                <Text>Vanderbilt University</Text>
                <Link color='blue.500' href='maddox.r.barron@vanderbilt.edu'>
                maddox.r.barron@vanderbilt.edu
                </Link>
              </CardBody>
            </Card>
            <Card maxWidth="400px">
              <CardHeader fontWeight="bold" fontSize="xl">Aditya Shrey</CardHeader>
              <CardBody>
                <Text>Vanderbilt University</Text>
                <Link color='blue.500' href='aditya.shrey@vanderbilt.edu'>
                aditya.shrey@vanderbilt.edu
                </Link>
              </CardBody>
            </Card>
            </SimpleGrid>
            <Text>
            We welcome the opportunity to discuss our project further and look forward to any inquiries.
            </Text>
          </VStack>
        </Flex>
      </Box>
    </ChakraProvider>
  );
}

export default Info;
