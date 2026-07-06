import invariant from 'invariant';
import { useParams } from '../../../common/utils/RoutingUtils';
import { ExperimentImproveView } from './ExperimentImproveView';

const ExperimentImprovePage = () => {
  const { experimentId } = useParams();
  invariant(experimentId, 'Experiment ID must be defined');

  return <ExperimentImproveView experimentId={experimentId} />;
};

export default ExperimentImprovePage;
