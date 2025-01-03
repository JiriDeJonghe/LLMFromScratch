#include <stdio.h>
#include "perceptron.h"

/**
 * @brief Creates an example dataset that can be used for training the model
 *        Defaults to creating a dataset to solve the AND gate problem
 *
 * @return Dataset* Pointer to the dataset that has been created
 */
Dataset* create_example_dataset() {
    const int NUM_SAMPLES = 100;
    const int NUM_INPUTS = 2;

    Dataset* dataset = create_dataset(NUM_SAMPLES, NUM_INPUTS);
    if (dataset == NULL) {
        return NULL;
    }

   float inputs[100][2] = {
        {1.169397139419055f, 13.81043565827701f},
        {6.938631834955604f, 15.584932585017079f},
        {17.303192323949517f, 0.9445826001125157f},
        {0.518789997982795f, 1.491224023408828f},
        {4.9421463664185366f, 12.096890778050575f},
        {1.2185581606890432f, 14.247000053464486f},
        {7.254525455150804f, 3.0154909634325433f},
        {15.9338470709928f, 12.295271490015722f},
        {7.394745437685092f, 3.017019760320696f},
        {14.006190629246369f, 9.234644384999847f},
        {1.669702910702413f, 19.605265576945182f},
        {20.81770761583412f, 16.870383156136135f},
        {3.1977724620419457f, 13.838295671058336f},
        {4.2749918774279605f, 9.255978556553616f},
        {11.902309195827739f, 18.335472411320268f},
        {20.728791469397365f, 14.651349746165522f},
        {18.214652052290745f, 0.04201001785680614f},
        {23.291643008379783f, 13.458584999533878f},
        {18.223261266853267f, 11.574920812359998f},
        {9.222000770163106f, 9.580110511929838f},
        {9.80923990664958f, 8.72238275017448f},
        {17.12065026922749f, 7.2661891081435055f},
        {6.49607454415521f, 5.6828654436303605f},
        {20.498468927792256f, 15.440510010606264f},
        {21.915935270683335f, 14.316374712441318f},
        {18.258145671950547f, 3.256618294933493f},
        {12.400044906193589f, 15.214155459883472f},
        {4.028058365316176f, 15.71656337275752f},
        {7.168614998612776f, 14.772131937376125f},
        {6.814633080735212f, 4.576030729257699f},
        {7.853101838825543f, 6.6042296269336624f},
        {11.043368011031426f, 5.522644417104869f},
        {13.064840626983303f, 17.951968159500183f},
        {5.217620874233327f, 18.50245185244278f},
        {19.196839846535436f, 12.371511304581093f},
        {17.44042678311685f, 2.7860716732793f},
        {20.383025121389206f, 2.750581508331602f},
        {6.420523309969735f, 7.317359827507202f},
        {14.755793067562431f, 2.309390754772054f},
        {15.666145756655737f, 10.236653580706177f},
        {9.283594419222855f, 5.187123229887137f},
        {9.848500285747651f, 2.209557844387222f},
        {19.44378118782329f, 3.76228106143053f},
        {14.408862369171237f, 6.479183324202036f},
        {21.560168126032664f, 13.310782865521261f},
        {11.760773401176433f, 6.239183440015683f},
        {19.879635173190763f, 10.478975092296498f},
        {20.61334730803773f, 6.67536834600611f},
        {18.92268344738623f, 1.7666972937182446f},
        {21.30373049297287f, 1.765715096242253f},
        {16.772658934859717f, 5.045857501704864f},
        {10.90536738265799f, 4.3651789026425325f},
        {7.341963954932532f, 1.2537837880263836f},
        {19.682025718222242f, 15.555541408728207f},
        {3.351951730176479f, 9.922642628910252f},
        {9.631115963536526f, 5.485012417663393f},
        {5.549095558568056f, 16.196666097349787f},
        {14.170986325040557f, 6.75352239708825f},
        {21.2704333904435f, 3.9122078325789267f},
        {7.865798169900561f, 6.839399065754712f},
        {19.31063164847155f, 13.455396909453494f},
        {19.138868147231555f, 5.662288871039718f},
        {15.976647546580676f, 5.313113205690416f},
        {5.858699921346563f, 17.62282346990982f},
        {19.664549898109527f, 10.78259858509967f},
        {11.222428570754776f, 14.875302699865983f},
        {17.738609447618146f, 17.95617014970672f},
        {11.05551523974087f, 5.47922184178182f},
        {20.738057989751383f, 10.668710403666278f},
        {16.85194300809872f, 2.663868838773744f},
        {4.031401762390966f, 19.237103815489448f},
        {4.301915355004047f, 16.249118551475682f},
        {20.39772068520218f, 13.028179965720039f},
        {8.76269392420765f, 4.045149011542184f},
        {9.144843345861041f, 19.416456058165167f},
        {0.988979766387379f, 0.7107207657266268f},
        {15.219913845953828f, 8.464684525717628f},
        {19.30028021868313f, 18.888284853628324f},
        {21.535509543384105f, 6.384194917098538f},
        {4.674323580690159f, 15.538285941575353f},
        {20.576216649691794f, 19.50183200019739f},
        {4.241509505537404f, 14.038691810716903f},
        {8.964506367608317f, 6.050581461318034f},
        {9.174690265369252f, 19.373278094009518f},
        {13.695480528372553f, 7.91426385957291f},
        {12.267281196988385f, 19.02618126671055f},
        {12.697855409827074f, 2.588120753775438f},
        {12.556864776263318f, 18.741283411054404f},
        {22.36506138223216f, 15.142160615362172f},
        {5.917822416166629f, 0.9223856320810708f},
        {4.999273224072193f, 13.257442896867666f},
        {19.584729362170727f, 4.531424582933523f},
        {7.292807873168513f, 10.389012711397687f},
        {16.2094173819089f, 10.213505930039279f},
        {22.16779315786074f, 2.1508385527033425f},
        {1.679266072710262f, 16.301268503581472f},
        {18.76246662245994f, 19.249060341646288f},
        {0.8424717364683127f, 11.77764785999159f},
        {12.217911903347067f, 8.085041013294504f},
        {10.53100016219056f, 13.117539847625217f},
    };
            
    // The output values in the case of AND
    int outputs[100] = {
        0,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        0,
        1,
        0,
        1,
        1,
    };


    // The output values in the case of XOR
    int outputs_xor[100] = {
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
    };

    for (int i = 0; i < NUM_SAMPLES; i++) {
        for (int j = 0; j < NUM_INPUTS; j++) {
            dataset->samples[i].inputs[j] = inputs[i][j];
        }
        dataset->samples[i].output = outputs_xor [i];
    }

    return dataset;
}


